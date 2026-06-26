"""
Live camera + Fanar-Oryx labeling — see what the camera sees and how well Oryx reads it.

  python oryx_view.py

Controls (focus the window):
  SPACE = detect once with Fanar-Oryx (boxes + labels drawn, also printed in terminal)
  A     = toggle AUTO mode (re-detect every few seconds)   <-- "live" labeling
  C     = clear boxes
  Q     = quit

NOTE: each detection is a Fanar API call. AUTO mode calls it every ORYX_INTERVAL
seconds (default 6) — mind your rate limit. Run this standalone (not while the
server is also using the camera).
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on path
from agent.agent3_vision import locate_scene
from tools import PRODUCTS

# OpenCV can't render Arabic, so show a short ASCII tag above each box (derived from the
# registry description, e.g. "hair serum — Kerastase ..." -> "hair serum"). Terminal keeps Arabic.
EN = {k: v.split("—")[0].split("-")[0].strip()[:20] for k, v in PRODUCTS.items()}

idx = int(os.environ.get("BASEER_CAM_INDEX", "0"))
interval = float(os.environ.get("ORYX_INTERVAL", "6"))
cap = cv2.VideoCapture(idx)
if not cap.isOpened():
    raise SystemExit(f"camera {idx} not available")

print("SPACE = detect once | A = auto | C = clear | Q = quit")
auto = False
last_items = []
last_t = 0.0


def detect(frame):
    h, w = frame.shape[:2]
    _ok, buf = cv2.imencode(".jpg", frame)
    try:
        items = locate_scene(buf.tobytes(), w, h, PRODUCTS)   # map to your category names
        if items:
            print("Fanar-Oryx:")
            for i, it in enumerate(items):
                print(f"  #{i + 1}: {it['label']}")
        else:
            print("Fanar-Oryx: none")
        return items
    except Exception as e:
        print("error:", e)
        return last_items


while True:
    ok, frame = cap.read()
    if not ok:
        continue
    now = time.time()
    if auto and now - last_t > interval:
        last_items = detect(frame)
        last_t = now

    disp = frame.copy()
    for i, it in enumerate(last_items):   # draw box + English tag above it (Arabic also in terminal)
        x1, y1, x2, y2 = it["box"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 3)
        tag = f"#{i + 1} {EN.get(it['label'], it['label'])}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(y1 - 10, th + 6)
        cv2.rectangle(disp, (x1, ty - th - 6), (x1 + tw + 6, ty + 4), (0, 0, 255), -1)  # bg for legibility
        cv2.putText(disp, tag, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    mode = "AUTO" if auto else "MANUAL"
    cv2.putText(disp, f"{mode}  items:{len(last_items)}  (SPACE/A/C/Q)", (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
    cv2.imshow("Baseer - Fanar-Oryx live", disp)

    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("c"):
        last_items = []
    if k == ord("a"):
        auto = not auto
        print("AUTO", "on" if auto else "off")
    if k == ord(" "):
        last_items = detect(frame)
        last_t = time.time()

cap.release()
cv2.destroyAllWindows()
