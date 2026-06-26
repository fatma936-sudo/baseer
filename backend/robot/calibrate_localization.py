"""
Calibrate a pixel -> hover-pose map so the arm can pre-position ABOVE an object that
Fanar-Oryx locates in the camera image. This is the "localization" layer: it learns,
directly from the fixed camera + fixed table, where to move the arm for any object
pixel — no camera extrinsics or IK tuning needed.

SIMPLE FLOW (no objects, no clicking): you drive the FOLLOWER with the LEADER arm.
The blue gripper TIP is auto-tracked (green dot). For ~12 spots across the table:
  - teleop the gripper down to hover just above a spot on the table,
  - check the green dot is on the gripper tip,
  - press 's'  (captures: that pixel + the arm's joints).
Spread the spots over the WHOLE table (left/center/right edges, near/far, and the spots
you'll actually place serums). Also press 'h' ONCE with the arm raised high to set the
'travel' pose (used for the top-down approach). Press 'q' to fit + save.

We fit a smooth map (pixel -> 6 joint angles) -> localization_map.json. At runtime Oryx
finds the target object's pixel and this map sends the arm there.

(If the green dot isn't on the tip, just CLICK the right spot — a click overrides the
auto-detection for that capture.)

    python backend/robot/calibrate_localization.py \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100 \
        --leader-port /dev/tty.usbmodem5AB90674941 --leader-id leader_so100 --cam 0

Controls (focus the camera window):
  drive the LEADER arm  = the follower mirrors it (teleoperation)
  s = capture a spot (auto gripper tip, or your click)   h = set the raised travel pose
  u = undo last        q = finish + fit + save (need >= 8 samples)
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

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "localization_map.json")  # backend/
KEYS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
        "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
W, H = 640, 480

click = {"uv": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click["uv"] = (x, y)


def feats(u, v):
    """Quadratic feature vector for normalized pixel (u,v) in [0,1]."""
    return np.array([1.0, u, v, u * u, u * v, v * v], dtype=np.float64)


def detect_gripper_tip(bgr):
    """Auto-locate the (blue) gripper tip in the frame so you don't have to click.
    Returns (u, v) of the lowest point of the largest blue blob, or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (90, 60, 40), (135, 255, 255))   # blue gripper
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 150:
        return None
    pts = c.reshape(-1, 2)
    tip = pts[pts[:, 1].argmax()]            # lowest point = finger tip (gripper points down)
    return (int(tip[0]), int(tip[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="follower port")
    ap.add_argument("--id", default="follower_so100")
    ap.add_argument("--leader-port", required=True, help="leader (teleop) port")
    ap.add_argument("--leader-id", default="leader_so100")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    cams = {"front": OpenCVCameraConfig(index_or_path=args.cam, width=W, height=H, fps=30)}
    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, cameras=cams))
    leader = SO100Leader(SO100LeaderConfig(port=args.leader_port, id=args.leader_id))
    robot.connect()
    leader.connect()
    cv2.namedWindow("localization calib")
    cv2.setMouseCallback("localization calib", on_mouse)
    print("Teleop ON: move the LEADER arm to drive the follower.")
    print("For each spot: CLICK the object, hover the arm above it, press 's'.  q=fit")

    samples = []   # list of (u_px, v_px, [6 joints])
    travel = None  # one raised "travel" pose for the top-down approach (press 'h')
    try:
        while True:
            action = leader.get_action()          # teleoperate: leader drives follower
            robot.send_action(action)
            try:
                obs = robot.get_observation()     # transient camera timeouts shouldn't crash us
            except Exception as e:
                print(f"[calib] camera hiccup, skipping frame: {e}")
                time.sleep(0.05)
                continue
            img = np.asarray(obs["front"])
            disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            auto = detect_gripper_tip(disp)                 # auto-track the gripper tip
            if auto:
                cv2.circle(disp, auto, 7, (0, 255, 0), 2)   # green = auto-detected tip
                cv2.putText(disp, "tip", (auto[0] + 8, auto[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if click["uv"]:
                cv2.circle(disp, click["uv"], 6, (0, 0, 255), 2)   # red = manual override click
            for (u, v, _) in samples:
                cv2.circle(disp, (int(u), int(v)), 4, (255, 0, 0), -1)
            cv2.putText(disp, f"samples={len(samples)}  travel={'set' if travel else 'NO'}  "
                        "s=capture h=travel u=undo q=fit",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("localization calib", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("u") and samples:
                samples.pop(); print("undid last sample")
            if k == ord("h"):     # capture the raised "travel" pose (arm high, retracted, safe)
                travel = {j: round(float(obs[j]), 2) for j in KEYS}
                print(f"  travel pose set: lift={travel['shoulder_lift.pos']} elbow={travel['elbow_flex.pos']}")
            if k == ord("s"):
                uv = click["uv"] or auto      # prefer a manual click; else the auto tip
                if not uv:
                    print("no gripper detected and no click — move the gripper into view or click a spot")
                    continue
                joints = [float(obs[j]) for j in KEYS]
                samples.append((uv[0], uv[1], joints))
                print(f"  saved #{len(samples)}: px={uv} ({'click' if click['uv'] else 'auto'}) "
                      f"lift={round(joints[1],1)}")
                click["uv"] = None
            time.sleep(0.02)
    finally:
        # persist raw samples FIRST so a disconnect hiccup can never lose them
        if samples:
            raw = os.path.join(os.path.dirname(OUT), "localization_samples.json")
            json.dump(samples, open(raw, "w"))
            print(f"saved {len(samples)} raw samples -> {raw}")
        for name, dev in (("leader", leader), ("robot", robot)):
            try:
                dev.disconnect()
            except Exception as e:
                print(f"[calib] {name} disconnect warning (ignored): {e}")
        cv2.destroyAllWindows()

    if len(samples) < 8:
        print(f"only {len(samples)} samples — need >= 8 to fit. Nothing saved.")
        return

    # Fit (u,v)->joint for each joint via least squares on quadratic features.
    X = np.stack([feats(u / W, v / H) for (u, v, _) in samples])     # (N,6)
    Y = np.stack([j for (_, _, j) in samples])                       # (N,6 joints)
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)                   # (6 feats, 6 joints)
    pred = X @ coeffs
    rms = np.sqrt(((pred - Y) ** 2).mean(axis=0))
    print("\nfit residual RMS per joint (deg):", [round(float(r), 1) for r in rms])

    us = [u for (u, v, _) in samples]; vs = [v for (u, v, _) in samples]
    px_range = {"umin": min(us), "umax": max(us), "vmin": min(vs), "vmax": max(vs)}
    out = {"W": W, "H": H, "keys": KEYS, "coeffs": coeffs.tolist(),
           "n_samples": len(samples), "px_range": px_range}
    if travel:
        out["travel_pose"] = travel
    else:
        print("⚠ no travel pose captured (press 'h' next time) — top-down approach will be weaker.")
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}  ({len(samples)} samples, covered u[{px_range['umin']}-{px_range['umax']}] "
          f"v[{px_range['vmin']}-{px_range['vmax']}], travel={'yes' if travel else 'no'})")
    print("Now agent4_grasp.py can pre-position above an Oryx-located object before grasping.")


if __name__ == "__main__":
    main()
