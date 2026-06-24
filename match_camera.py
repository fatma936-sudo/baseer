"""
Match the live camera to the TRAINING viewpoint by overlaying a ghost of a
training frame on the live feed. Physically move the camera until the fixed
features (table edges, checkerboard, wall corner) line up with the ghost.

  python match_camera.py

Controls (focus the window):
  +/-  = more/less ghost (reference) opacity
  S    = save current live frame to ~/baseer/cam_now.jpg
  Q    = quit
"""
import os
import glob

import cv2

HOME = os.path.expanduser("~")
REF = os.path.join(HOME, "baseer", "ref_view.jpg")
DATASET = os.path.join(HOME, ".cache/huggingface/lerobot/55CancriE/baseer_vanity_20260621_193229")

# Load the reference; if missing, grab a frame straight from the training video.
ref = cv2.imread(REF) if os.path.exists(REF) else None
if ref is None:
    vids = glob.glob(DATASET + "/videos/observation.images.front/**/*.mp4", recursive=True)
    if vids:
        c = cv2.VideoCapture(sorted(vids)[0])
        c.set(cv2.CAP_PROP_POS_MSEC, 1000)  # 1s in
        ok, ref = c.read()
        c.release()
        if ok:
            cv2.imwrite(REF, ref)
            print("extracted reference from training video ->", REF)
if ref is None:
    raise SystemExit("No reference frame found. Put a training frame at ~/baseer/ref_view.jpg")
ref = cv2.resize(ref, (640, 480))

cap = cv2.VideoCapture(int(os.environ.get("BASEER_CAM_INDEX", "0")))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
alpha = 0.5
print("Align the LIVE camera so fixed features match the GHOST (training view).")
print("Match: table edges, checkerboard position, wall corner.  +/- opacity, S save, Q quit")

while True:
    ok, live = cap.read()
    if not ok:
        continue
    live = cv2.resize(live, (640, 480))
    blend = cv2.addWeighted(live, 1 - alpha, ref, alpha, 0)
    cv2.putText(blend, f"ghost={alpha:.1f}  (+/- , S=save , Q=quit)", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Match camera to training view", blend)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k in (ord("+"), ord("=")):
        alpha = min(0.9, alpha + 0.1)
    if k in (ord("-"), ord("_")):
        alpha = max(0.1, alpha - 0.1)
    if k == ord("s"):
        cv2.imwrite(os.path.join(HOME, "baseer", "cam_now.jpg"), live)
        print("saved current live frame -> ~/baseer/cam_now.jpg")

cap.release()
cv2.destroyAllWindows()
