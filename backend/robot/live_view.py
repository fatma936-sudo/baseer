"""
Live camera preview + on-demand Fanar-Oryx perception.

  python live_view.py

Controls (focus the camera window):
  SPACE = run perception on the current frame  (result prints in the terminal)
  S     = save the current frame to vanity.jpg
  Q     = quit

Note: OpenCV windows can't render Arabic, so detection results print in the
terminal (Arabic shows correctly there); the window shows the live feed + status.
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on path
from agent.agent3_vision import describe_scene
from tools import SCENE_ITEMS

idx = int(os.environ.get("BASEER_CAM_INDEX", "0"))
cap = cv2.VideoCapture(idx)
if not cap.isOpened():
    raise SystemExit(f"camera {idx} not available")

print("Live view — SPACE: perceive | S: save frame | Q: quit")
print("catalog:", SCENE_ITEMS)
status = "SPACE = perceive"

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    disp = frame.copy()
    cv2.putText(disp, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    cv2.imshow("Baseer camera (SPACE=perceive  S=save  Q=quit)", disp)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("s"):
        cv2.imwrite("vanity.jpg", frame)
        print("saved vanity.jpg")
        status = "saved vanity.jpg"
    if k == ord(" "):
        status = "perceiving..."
        _ok, buf = cv2.imencode(".jpg", frame)
        try:
            items = describe_scene(buf.tobytes(), SCENE_ITEMS)
            print("present:", items if items else "none")
            status = "present: " + str(len(items)) + " item(s) (see terminal)"
        except Exception as e:
            print("error:", e)
            status = "error (see terminal)"

cap.release()
cv2.destroyAllWindows()
