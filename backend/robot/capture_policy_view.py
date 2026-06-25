"""
Capture EXACTLY what the policy sees through the robot's camera (same code path as
grasp.py) and save it next to a TRAINING frame, so we can confirm the deploy view
matches what the model was trained on. A mismatch (wrong camera, black frame,
wrong colors, different angle/zoom) makes the policy grasp empty air.

    /opt/anaconda3/envs/lerobot/bin/python capture_policy_view.py \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100 --cam 0

Writes:
  ~/baseer/deploy_view.jpg  <- live frame the policy receives RIGHT NOW
  ~/baseer/train_view.jpg   <- a frame from the training dataset video
Open both and compare angle / zoom / lighting / where the serum sits.
"""
import argparse
import glob
import os

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

HOME = os.path.expanduser("~")
DATASET = os.path.join(HOME, ".cache/huggingface/lerobot/55CancriE/baseer_serums_20260623_204039")


def save_train_frame():
    vids = sorted(glob.glob(DATASET + "/videos/observation.images.front/**/*.mp4", recursive=True))
    if not vids:
        print("no training video found to compare against")
        return
    c = cv2.VideoCapture(vids[0])
    c.set(cv2.CAP_PROP_POS_MSEC, 3000)   # 3s in (arm mid-reach)
    ok, frame = c.read()
    c.release()
    if ok:
        out = os.path.join(HOME, "baseer", "train_view.jpg")
        cv2.imwrite(out, frame)
        print(f"train frame  -> {out}  shape={frame.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default="follower_so100")
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    cams = {"front": OpenCVCameraConfig(index_or_path=args.cam, width=640, height=480, fps=30)}
    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, cameras=cams))
    robot.connect()
    # warm up, then grab the exact observation the policy gets
    obs = None
    for _ in range(10):
        obs = robot.get_observation()
    robot.disconnect()

    img = obs.get("front")
    if img is None:
        print("‼ NO 'front' image in observation — camera capture FAILED")
        return
    img = np.asarray(img)
    print(f"deploy frame shape={img.shape} dtype={img.dtype} "
          f"min={img.min()} max={img.max()} mean={img.mean():.1f}")
    if img.max() == img.min():
        print("‼ frame is a SOLID color (all pixels equal) — camera not delivering real video")
    # lerobot delivers RGB; OpenCV writes BGR, so convert for a correct-looking jpg
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 and img.shape[2] == 3 else img
    out = os.path.join(HOME, "baseer", "deploy_view.jpg")
    cv2.imwrite(out, bgr)
    print(f"deploy frame -> {out}")
    save_train_frame()
    print("\nNow open BOTH and compare:  deploy_view.jpg  vs  train_view.jpg")


if __name__ == "__main__":
    main()
