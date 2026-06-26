"""
Record a few FIXED SLOTS for reliable type-selection (no fuzzy localization map).

For each slot you (1) place a serum there, (2) CLICK it in the camera window (where Oryx
will see it), (3) teleop so the OPEN gripper STRADDLES the object at grasp height — i.e.
the exact position where simply CLOSING the gripper would grab it (fingers around the
bottle, not above it) — (4) press 's'. Do this for each spot a serum will sit (2-3 slots).
Also press 'h' once with the arm raised to set the high 'travel' pose (clears obstacles).

At runtime: Oryx finds the requested serum's pixel -> snap to the NEAREST recorded slot ->
move to that exact pose -> CLOSE the gripper (scripted) -> verify -> deliver. Because the
slot IS the grasp pose, this is precise and reliable (no fuzzy map, no policy drift).

    python backend/robot/record_slots.py \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100 \
        --leader-port /dev/tty.usbmodem5AB90674941 --leader-id leader_so100 --cam 0

Controls (focus the camera window):
  drive the LEADER  = follower mirrors it
  click the object  = mark the slot's pixel (do this before 's')
  s = save a slot (clicked pixel + current pose)   h = set the raised travel pose
  u = undo last slot     d = done + save     q = quit without saving
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "slots.json")  # backend/
KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
W, H = 640, 480
click = {"uv": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["uv"] = (x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default="follower_so100")
    ap.add_argument("--leader-port", required=True)
    ap.add_argument("--leader-id", default="leader_so100")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    cams = {"front": OpenCVCameraConfig(index_or_path=args.cam, width=W, height=H, fps=30)}
    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, cameras=cams))
    leader = SO100Leader(SO100LeaderConfig(port=args.leader_port, id=args.leader_id))
    robot.connect(); leader.connect()
    cv2.namedWindow("record slots"); cv2.setMouseCallback("record slots", on_mouse)
    print("Teleop ON. Per slot: place serum, CLICK it, put the OPEN gripper AROUND it at")
    print("grasp height (where closing would grab it), press 's'. 'h'=travel pose, 'd'=save.")

    slots = []      # [{"pixel":[u,v], "pose":{...}}]
    travel = None
    try:
        while True:
            robot.send_action(leader.get_action())
            try:
                obs = robot.get_observation()
            except Exception as e:
                print(f"[slots] camera hiccup: {e}"); time.sleep(0.05); continue
            disp = cv2.cvtColor(np.asarray(obs["front"]), cv2.COLOR_RGB2BGR)
            if click["uv"]:
                cv2.circle(disp, click["uv"], 6, (0, 0, 255), 2)
            for i, s in enumerate(slots):
                u, v = s["pixel"]
                cv2.circle(disp, (u, v), 6, (0, 255, 0), -1)
                cv2.putText(disp, str(i + 1), (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp, f"slots={len(slots)} travel={'set' if travel else 'NO'}  "
                        "click+s | h | u | d=save", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("record slots", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                slots = []; break
            if k == ord("d"):
                break
            if k == ord("u") and slots:
                slots.pop(); print("undid last slot")
            if k == ord("h"):
                travel = {j: round(float(obs[j]), 2) for j in KEYS}
                print(f"  travel pose set (lift={travel['shoulder_lift.pos']})")
            if k == ord("s"):
                if not click["uv"]:
                    print("click the object first!"); continue
                pose = {j: round(float(obs[j]), 2) for j in KEYS}
                slots.append({"pixel": list(click["uv"]), "pose": pose})
                json.dump({"slots": slots, "travel_pose": travel}, open(OUT, "w"), indent=2)  # save now
                print(f"  saved slot #{len(slots)} at pixel {click['uv']}")
                click["uv"] = None
            time.sleep(0.02)
    finally:
        for name, dev in (("leader", leader), ("robot", robot)):
            try:
                dev.disconnect()
            except Exception as e:
                print(f"[slots] {name} disconnect warning: {e}")
        cv2.destroyAllWindows()

    if not slots:
        print("No slots saved."); return
    if travel is None:
        print("⚠ no travel pose ('h') — obstacle clearance will be weaker.")
    json.dump({"slots": slots, "travel_pose": travel}, open(OUT, "w"), indent=2)
    print(f"\nwrote {len(slots)} slots -> {OUT}")


if __name__ == "__main__":
    main()
