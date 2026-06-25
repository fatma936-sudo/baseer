"""
Record the fixed DELIVERY trajectory once, so the arm can hand the object to the
user deterministically (the delivery zone never moves — no need to make the policy
learn the boring carry).

Torque is DISABLED so you can move the follower BY HAND. You snap a few keyframes:
typically  (1) lift straight up,  (2) swing toward the user,  (3) the hand-off pose.
grasp.py then replays these in order after a verified grasp and opens the gripper.

Run (arm powered + calibrated; no policy needed):

    /opt/anaconda3/envs/lerobot/bin/python save_delivery_pose.py \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100

Then: move the arm to each pose, press Enter to capture it. Type 'd' + Enter when done.
Writes delivery_pose.json (a list of waypoints).
"""
import argparse
import json
import os

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "delivery_pose.json")  # backend/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default="follower_so100")
    args = ap.parse_args()

    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, cameras={}))
    robot.connect()
    robot.bus.disable_torque()          # let you move it by hand
    print("Torque OFF — move the arm by hand.\n")
    print("Move to a waypoint, press Enter to capture it. Capture them in ORDER")
    print("(e.g. lift up -> swing to user -> hand-off). Type 'd' then Enter when done.\n")

    waypoints = []
    try:
        while True:
            cmd = input(f"[{len(waypoints)} captured] Enter=capture, d=done: ").strip().lower()
            if cmd == "d":
                break
            pose = {k: round(float(v), 2)
                    for k, v in robot.get_observation().items() if k.endswith(".pos")}
            waypoints.append(pose)
            print(f"  captured #{len(waypoints)}: {pose}")
    finally:
        robot.bus.enable_torque()
        robot.disconnect()

    if not waypoints:
        print("No waypoints captured — nothing written.")
        return
    json.dump(waypoints, open(OUT, "w"), indent=2)
    print(f"\nwrote {len(waypoints)} waypoints -> {OUT}")


if __name__ == "__main__":
    main()
