"""
Test ANY YOLO-World classes on an image or the live camera — no training needed.
YOLO-World is open-vocabulary: a "class" is just a text prompt.

Usage:
  # on an image:
  python yolo_test.py "perfume bottle,red lipstick,hair brush" photo.jpg
  # on the live camera (index 0):
  python yolo_test.py "perfume bottle,red lipstick,hair brush"
  # tune sensitivity:
  CONF=0.02 python yolo_test.py "tiny ring,gold earring" photo.jpg

Prints detections and saves an annotated image -> yolo_test_out.jpg
"""
import os
import sys

import cv2
from ultralytics import YOLOWorld

DEFAULT = "perfume bottle,cosmetic cream jar,sunscreen bottle"


def main():
    classes = [c.strip() for c in (sys.argv[1] if len(sys.argv) > 1 else DEFAULT).split(",") if c.strip()]
    src = sys.argv[2] if len(sys.argv) > 2 else None
    conf = float(os.environ.get("CONF", "0.05"))

    model = YOLOWorld(os.environ.get("BASEER_YOLO_MODEL", "yolov8s-worldv2.pt"))
    model.set_classes(classes)

    if src is None:  # live camera
        cap = cv2.VideoCapture(int(os.environ.get("BASEER_CAM_INDEX", "0")))
        frame = None
        for _ in range(5):
            ok, frame = cap.read()
        cap.release()
        if frame is None:
            print("camera not available (grant Camera permission / check index)")
            return
        img = frame
    else:
        img = src

    res = model.predict(img, conf=conf, verbose=False)[0]
    print("classes :", classes)
    dets = []
    if res.boxes is not None:
        for c, cf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            dets.append(f"{classes[int(c)]} ({cf:.2f})")
    print("detected:", dets or "none (try lowering CONF or rephrasing prompts)")

    cv2.imwrite("yolo_test_out.jpg", res.plot())
    print("annotated image -> yolo_test_out.jpg")


if __name__ == "__main__":
    main()
