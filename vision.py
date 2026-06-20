"""
YOLO-World perception for `perceive_scene`: open-vocabulary detection of the
vanity items on the live camera — no training required.

We give YOLO-World English text prompts per catalog item and map detections
back to the Arabic catalog name. Tune BASEER_YOLO_CONF if it over/under-detects.

    python vision.py              # detect on the live camera (index 0)
    python vision.py <image>      # detect on an image file
"""
import os
import threading

import cv2

# Arabic catalog name -> open-vocabulary English prompts to detect it
CATALOG = {
    "عطر ديور": ["perfume bottle", "perfume", "cologne bottle", "glass perfume bottle"],
    "كريم مرطب": ["cosmetic cream jar", "face cream jar", "moisturizer", "lotion jar"],
    "واقي شمس": ["sunscreen bottle", "sunscreen tube", "lotion bottle"],
}

CONF = float(os.environ.get("BASEER_YOLO_CONF", "0.05"))
MODEL_NAME = os.environ.get("BASEER_YOLO_MODEL", "yolov8s-worldv2.pt")
CAM_INDEX = int(os.environ.get("BASEER_CAM_INDEX", "0"))

_model = None
_idx2ar = None
_lock = threading.Lock()


def _device():
    try:
        import torch
        if os.environ.get("BASEER_YOLO_DEVICE"):
            return os.environ["BASEER_YOLO_DEVICE"]
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _build_classes():
    classes, idx2ar = [], {}
    for ar, prompts in CATALOG.items():
        for p in prompts:
            idx2ar[len(classes)] = ar
            classes.append(p)
    return classes, idx2ar


def get_model():
    global _model, _idx2ar
    with _lock:
        if _model is None:
            from ultralytics import YOLOWorld
            m = YOLOWorld(MODEL_NAME)
            classes, _idx2ar = _build_classes()
            m.set_classes(classes)
            _model = m
    return _model


def detect(image, conf=CONF):
    """Return the catalog items (Arabic) present in `image` (path or BGR array)."""
    m = get_model()
    results = m.predict(image, conf=conf, device=_device(), verbose=False)
    found = set()
    for r in results:
        if r.boxes is None:
            continue
        for c in r.boxes.cls.tolist():
            found.add(_idx2ar[int(c)])
    return [ar for ar in CATALOG if ar in found]  # preserve catalog order


def capture_frame(index=CAM_INDEX):
    cap = cv2.VideoCapture(index)
    frame = None
    try:
        for _ in range(5):  # warm up
            ok, f = cap.read()
            if ok:
                frame = f
    finally:
        cap.release()
    if frame is None:
        raise RuntimeError(f"camera {index} not available")
    return frame


def perceive(index=CAM_INDEX, conf=CONF):
    """Grab a live frame and return the detected catalog items."""
    return detect(capture_frame(index), conf)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else None
    items = detect(src) if src else perceive()
    print("detected:", items)
