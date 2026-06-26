"""
Record the fixed DELIVERY trajectory (teleoperation) so the arm can hand the object to
the user deterministically after a verified grasp. agent4_grasp.deliver_to_zone() replays
these waypoints (gripper held CLOSED during the carry) then opens at the last one.

Drive the FOLLOWER with the LEADER arm (same as recording) and capture a FEW DISTINCT
waypoints IN ORDER — MOVE the arm between each capture, e.g.:
   1) lift straight UP off the table
   2) swing toward the user / delivery zone
   3) (optional) lower a bit at the zone   <- last waypoint = where it releases

    python backend/robot/save_delivery_pose.py \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100 \
        --leader-port /dev/tty.usbmodem5AB90674941 --leader-id leader_so100 --cam 0

Controls (focus the camera window):
  drive the LEADER  = follower mirrors it
  s  = capture the current pose as the next waypoint
  u  = undo last waypoint
  d  = done (save)     q = quit without saving
IMPORTANT: move the arm to a NEW pose before each 's' — identical waypoints = no motion.
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

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "delivery_pose.json")  # backend/
KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="follower port")
    ap.add_argument("--id", default="follower_so100")
    ap.add_argument("--leader-port", required=True, help="leader (teleop) port")
    ap.add_argument("--leader-id", default="leader_so100")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    cams = {"front": OpenCVCameraConfig(index_or_path=args.cam, width=640, height=480, fps=30)}
    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, cameras=cams))
    leader = SO100Leader(SO100LeaderConfig(port=args.leader_port, id=args.leader_id))
    robot.connect()
    leader.connect()
    cv2.namedWindow("save delivery pose")
    print("Teleop ON. Move the arm to each waypoint (lift -> swing -> zone), press 's'.")
    print("MOVE between captures! 'd'=done, 'u'=undo, 'q'=quit.")

    waypoints = []
    try:
        while True:
            action = leader.get_action()
            robot.send_action(action)
            try:
                obs = robot.get_observation()
            except Exception as e:
                print(f"[delivery] camera hiccup, skipping frame: {e}"); time.sleep(0.05); continue
            disp = cv2.cvtColor(np.asarray(obs["front"]), cv2.COLOR_RGB2BGR)
            cv2.putText(disp, f"waypoints={len(waypoints)}  s=capture u=undo d=done q=quit",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("save delivery pose", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                waypoints = []; break
            if k == ord("d"):
                break
            if k == ord("u") and waypoints:
                waypoints.pop(); print("undid last waypoint")
            if k == ord("s"):
                pose = {key: round(float(obs[key]), 2) for key in KEYS}
                waypoints.append(pose)
                json.dump(waypoints, open(OUT, "w"), indent=2)   # save NOW, never lose on a hiccup
                print(f"  captured #{len(waypoints)} (saved): shoulder_lift={pose['shoulder_lift.pos']} "
                      f"pan={pose['shoulder_pan.pos']} elbow={pose['elbow_flex.pos']}")
            time.sleep(0.02)
    finally:
        for name, dev in (("leader", leader), ("robot", robot)):
            try:
                dev.disconnect()
            except Exception as e:
                print(f"[delivery] {name} disconnect warning (ignored): {e}")
        cv2.destroyAllWindows()

    if not waypoints:
        print("No waypoints saved.")
        return
    # warn if waypoints barely differ (the bug we just hit)
    lifts = [w["shoulder_lift.pos"] for w in waypoints]
    if len(waypoints) >= 2 and (max(lifts) - min(lifts)) < 5:
        print("⚠ WARNING: waypoints barely differ in height — did you move the arm between captures?")
    json.dump(waypoints, open(OUT, "w"), indent=2)
    print(f"\nwrote {len(waypoints)} waypoints -> {OUT}")


if __name__ == "__main__":
    main()
