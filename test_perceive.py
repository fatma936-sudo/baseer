"""
Test Fanar-Oryx perception (perceive_scene) on the live camera or an image.

  python test_perceive.py             # grab a live camera frame
  python test_perceive.py vanity.jpg  # use an image file

Prints the catalog and which items Fanar-Oryx sees present (reads labels).
"""
import sys

import cv2

from fanar import describe_scene
from tools import SCENE_ITEMS

src = sys.argv[1] if len(sys.argv) > 1 else None
if src:
    image_bytes = open(src, "rb").read()
    print(f"image  : {src}")
else:
    import vision
    _ok, buf = cv2.imencode(".jpg", vision.capture_frame())
    image_bytes = buf.tobytes()
    print("image  : live camera frame")

print("catalog:", SCENE_ITEMS)
print("present:", describe_scene(image_bytes, SCENE_ITEMS))
