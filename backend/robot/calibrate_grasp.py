"""
Calibrate the grasp-success thresholds for the SO-100 gripper.

WHY: a "smart" retry needs to KNOW whether it actually caught the object. We read
two Feetech signals from the gripper motor (id 6):
  - Present_Position  (0..100 here)  -> an empty gripper closes ALL the way; an
                                        object holds the fingers open at its width.
  - Present_Current   (torque proxy) -> pressing on an object keeps current high;
                                        an empty closed gripper settles low.

This script closes the gripper TWICE — once on nothing, once on the object you
hold between the fingers — and writes the measured thresholds to grasp_cfg.json,
which grasp.py then uses to decide success/failure.

Run (no trained policy needed, just the arm powered + calibrated):

    /opt/anaconda3/envs/lerobot/bin/python calibrate_grasp.py \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100

Follow the prompts. Hold the HAIR SERUM in the gripper when asked.
"""
import argparse
import json
import os
import time

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "grasp_cfg.json")  # backend/


def _read(robot, key):
    """Return the gripper-motor value for a Feetech control-table field."""
    return float(robot.bus.sync_read(key)["gripper"])


def _set_gripper(robot, target):
    """Command only the gripper to `target` (0..100), holding the arm where it is."""
    obs = robot.get_observation()
    action = {k: v for k, v in obs.items() if k.endswith(".pos")}
    action["gripper.pos"] = float(target)
    robot.send_action(action)


def _sample(robot, secs=1.2, hz=30):
    """Drive a fixed close, then sample settled pos + current."""
    pos, cur = [], []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < secs:
        # keep commanding full close so the motor presses against whatever's there
        _set_gripper(robot, 0.0)
        time.sleep(1.0 / hz)
        pos.append(_read(robot, "Present_Position"))
        cur.append(abs(_read(robot, "Present_Current")))
    # use the last third (settled)
    k = max(1, len(pos) // 3)
    settled_pos = sum(pos[-k:]) / k
    settled_cur = sum(cur[-k:]) / k
    return settled_pos, settled_cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default="follower_so100")
    args = ap.parse_args()

    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, cameras={}))
    robot.connect()
    print("connected.\n")

    input("1) EMPTY gripper — make sure nothing is between the fingers, then press Enter…")
    _set_gripper(robot, 60.0); time.sleep(1.0)          # open first
    empty_pos, empty_cur = _sample(robot)
    print(f"   empty:  pos={empty_pos:.1f}  current={empty_cur:.0f}\n")

    _set_gripper(robot, 60.0); time.sleep(1.0)          # re-open for loading
    input("2) HOLD the HAIR SERUM between the fingers, then press Enter…")
    held_pos, held_cur = _sample(robot)
    print(f"   holding: pos={held_pos:.1f}  current={held_cur:.0f}\n")

    _set_gripper(robot, 60.0); time.sleep(0.5)
    robot.disconnect()

    # Threshold = midpoint between empty and holding for each signal.
    if held_pos - empty_pos < 3:
        print("WARNING: gripper position barely differs empty vs holding — the object "
              "may be too thin. Current will carry the decision.")
    cfg = {
        "gripper_open_cmd": 60.0,
        "gripper_close_cmd": 0.0,
        "empty_pos": round(empty_pos, 1),
        "held_pos": round(held_pos, 1),
        "pos_threshold": round((empty_pos + held_pos) / 2, 1),
        "empty_current": round(empty_cur, 0),
        "held_current": round(held_cur, 0),
        "current_threshold": round((empty_cur + held_cur) / 2, 0),
    }
    with open(CFG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", CFG_PATH)
    print(json.dumps(cfg, indent=2))
    print("\nRule grasp.py will use:  grasp_ok = (pos > pos_threshold) AND "
          "(current > current_threshold)")


if __name__ == "__main__":
    main()
