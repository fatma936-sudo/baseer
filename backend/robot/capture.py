"""Grab a clean frame from the camera -> vanity.jpg (for testing perception)."""
import os
import cv2

idx = int(os.environ.get("BASEER_CAM_INDEX", "0"))
cap = cv2.VideoCapture(idx)
frame = None
for _ in range(8):           # warm up the camera
    ok, f = cap.read()
    if ok:
        frame = f
cap.release()
if frame is None:
    raise SystemExit(f"camera {idx} not available")
cv2.imwrite("vanity.jpg", frame)
print("saved vanity.jpg", frame.shape)
