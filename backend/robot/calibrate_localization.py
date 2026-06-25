"""
Calibrate a pixel -> hover-pose map so the arm can pre-position ABOVE an object that
Fanar-Oryx locates in the camera image. This is the "localization" layer: it learns,
directly from the fixed camera + fixed table, where to move the arm for any object
pixel — no camera extrinsics or IK tuning needed.

HOW IT WORKS: torque is OFF so you move the arm by hand. For ~12 spots across the
table you (1) CLICK the object's center in the camera window, then (2) hand-pose the
arm hovering just above that object in a ready-to-grasp pose and press 's' to capture
the joints. We then fit a smooth quadratic map (u,v) -> 6 joint angles and save it to
localization_map.json, which grasp.py uses to pre-position before the policy grasps.

    /opt/anaconda3/envs/lerobot/bin/python calibrate_localization.py \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100 --cam 0

Controls (focus the camera window):
  click   = mark the object's center (do this BEFORE pressing s)
  s       = save a sample (current click + current joints)
  u       = undo last sample
  q       = finish and fit the map  (need >= 8 samples)

Tip: spread the spots — near/far, left/right/center — to cover the whole reachable
table. Put the arm in the SAME top-down grasp orientation each time, just translated.
"""
import argparse
import json
import os

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default="follower_so100")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    cams = {"front": OpenCVCameraConfig(index_or_path=args.cam, width=W, height=H, fps=30)}
    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, cameras=cams))
    robot.connect()
    robot.bus.disable_torque()                 # move the arm by hand
    cv2.namedWindow("localization calib")
    cv2.setMouseCallback("localization calib", on_mouse)
    print("Torque OFF. For each spot: CLICK the object, hover the arm above it, press 's'.")

    samples = []   # list of (u_px, v_px, [6 joints])
    try:
        while True:
            obs = robot.get_observation()
            img = np.asarray(obs["front"])
            disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if click["uv"]:
                cv2.circle(disp, click["uv"], 6, (0, 0, 255), 2)
            for (u, v, _) in samples:
                cv2.circle(disp, (int(u), int(v)), 4, (0, 255, 0), -1)
            cv2.putText(disp, f"samples={len(samples)}  click obj, then 's'. q=fit",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("localization calib", disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("u") and samples:
                samples.pop(); print("undid last sample")
            if k == ord("s"):
                if not click["uv"]:
                    print("click the object first!"); continue
                joints = [float(obs[j]) for j in KEYS]
                u, v = click["uv"]
                samples.append((u, v, joints))
                print(f"  saved #{len(samples)}: px=({u},{v}) joints={[round(x,1) for x in joints]}")
                click["uv"] = None
    finally:
        robot.bus.enable_torque()
        robot.disconnect()
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

    json.dump({"W": W, "H": H, "keys": KEYS, "coeffs": coeffs.tolist(),
               "n_samples": len(samples)}, open(OUT, "w"), indent=2)
    print(f"wrote {OUT}  ({len(samples)} samples)")
    print("Now grasp.py can pre-position above an Oryx-located object before grasping.")


if __name__ == "__main__":
    main()
