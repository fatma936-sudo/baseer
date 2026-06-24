"""
The robot's entire action space — three tools.

Right now they're STUBS (print + return a JSON-serializable observation).
Later, swap the bodies for the real implementations, keeping the signatures:
    perceive_scene -> vision detector / Fanar vision
    deliver        -> the trained ACT policy
    say            -> Aura TTS

Edit SCENE_ITEMS to test grounding and the graceful-failure path
(e.g. ask for something not in the list).
"""

import os

# Product registry: catalog name -> how to recognize it (brand text AND color/shape).
# This is what lets Fanar-Oryx map "Living Proof Perfect Hair Day" -> dry shampoo, etc.,
# and fall back to color when the label is unreadable. Edit to match YOUR products.
PRODUCTS = {
    "عطر الياسمين":     "perfume — The Body Shop 'Wild Jasmine' (clear glass bottle, amber/yellow liquid)",
    "عطر شيا":          "perfume — The Body Shop 'Shea' (clear glass bottle)",
    "عطر المسك الأبيض": "perfume — The Body Shop 'White Musk' (clear glass bottle, clear liquid)",
    "شامبو جاف":        "dry shampoo — brand 'Living Proof Perfect Hair Day' (tall slim aerosol can)",
    "بودرة الشعر":      "hair styling powder spray — brand 'LA Biosthetique Style' Powder Spray (tall aerosol can)",
    "سيروم الشعر":      "hair serum — brand 'Kerastase' (small bottle/tube)",
    "سيروم الوجه":      "face night serum — brand 'Vichy' / 'Normaderm' (small bottle)",
}
# The two aerosol cans (شامبو جاف vs بودرة الشعر) look alike -> distinguished by BRAND text.
SCENE_ITEMS = list(PRODUCTS.keys())  # "السيروم" alone stays ambiguous (two serums) -> agent asks


def perceive_scene():
    """Look at the vanity and return the items currently present.

    Backend via BASEER_PERCEIVE: 'oryx' (Fanar vision, reads labels) | 'yolo' | 'stub'.
    """
    backend = os.environ.get(
        "BASEER_PERCEIVE", "yolo" if os.environ.get("BASEER_VISION") == "1" else "stub"
    ).lower()

    if backend == "oryx":
        try:
            import cv2, vision
            from fanar import describe_scene
            _ok, buf = cv2.imencode(".jpg", vision.capture_frame())
            items = describe_scene(buf.tobytes(), PRODUCTS)   # registry-grounded (brand + color)
            print(f"  [tool] perceive_scene() [Fanar-Oryx] -> {items}")
            return {"items": items}
        except Exception as e:
            print(f"  [tool] perceive_scene() oryx failed ({e}); using stub list")
    elif backend == "yolo":
        try:
            import vision
            items = vision.perceive()
            print(f"  [tool] perceive_scene() [YOLO] -> {items}")
            return {"items": items}
        except Exception as e:
            print(f"  [tool] perceive_scene() vision failed ({e}); using stub list")

    print(f"  [tool] perceive_scene() -> {SCENE_ITEMS}")
    return {"items": list(SCENE_ITEMS)}


# Per-item task strings the grasp policy was trained on (English single_task).
GRASP_TASKS = {
    "سيروم الشعر": "Pick up the hair serum and place it in the delivery zone",
    "سيروم الوجه": "Pick up the face serum and place it in the delivery zone",
}

_GRASP = None  # lazily-built, reused GraspController (keeps one robot connection)


def _grasp_controller():
    global _GRASP
    if _GRASP is None:
        from grasp import GraspController
        _GRASP = GraspController(
            policy_path=os.environ.get("BASEER_POLICY", "~/baseer/policy_vla/pretrained_model"),
            port=os.environ.get("BASEER_FOLLOWER_PORT", "/dev/tty.usbmodem5AB90677591"),
            robot_id=os.environ.get("BASEER_FOLLOWER_ID", "follower_so100"),
            cam_index=int(os.environ.get("BASEER_CAM_INDEX", "0")),
        )
    return _GRASP


def deliver(item):
    """Pick `item` from the vanity and place it at the fixed delivery zone.

    With BASEER_GRASP=policy, drives the real arm via the closed-loop, retrying
    GraspController (torque + width verification, re-approach on a miss). Otherwise
    stays a stub so the voice/agent stack runs without the arm.
    """
    if item not in SCENE_ITEMS:
        print(f"  [tool] deliver(item='{item}') -> NOT PRESENT")
        return {"ok": False, "error": "item_not_present", "item": item,
                "available": list(SCENE_ITEMS)}

    if os.environ.get("BASEER_GRASP") == "policy":
        task = GRASP_TASKS.get(item)
        if task is None:
            print(f"  [tool] deliver(item='{item}') -> no trained grasp policy for this item")
            return {"ok": False, "error": "no_policy_for_item", "item": item}
        try:
            attempts = int(os.environ.get("BASEER_GRASP_ATTEMPTS", "3"))
            ok = _grasp_controller().pick(task, attempts=attempts, item_name=item)
            print(f"  [tool] deliver(item='{item}') -> {'DELIVERED' if ok else 'FAILED after retries'}")
            return {"ok": ok, "item": item, "error": None if ok else "grasp_failed"}
        except Exception as e:
            print(f"  [tool] deliver(item='{item}') grasp error: {e}")
            return {"ok": False, "error": f"grasp_error:{e}", "item": item}

    print(f"  [tool] deliver(item='{item}') -> delivering to zone... (stub)")
    return {"ok": True, "item": item}


def say(text_ar):
    """Speak a short Arabic message to the user (their only feedback channel)."""
    print(f"  [SAY] 🔊 {text_ar}")
    return {"ok": True}


def ask(text_ar):
    """Ask the user a clarifying question (speaks it) and wait for their answer."""
    print(f"  [ASK] ❓ {text_ar}")
    return {"ok": True, "awaiting": True}


# --- dispatch + OpenAI-style tool schema -----------------------------------
DISPATCH = {"perceive_scene": perceive_scene, "deliver": deliver, "say": say, "ask": ask}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "perceive_scene",
            "description": "Look at the vanity and return the list of items currently present. "
                           "ALWAYS call this before delivering anything.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deliver",
            "description": "Pick ONE item from the vanity and place it at the user's fixed "
                           "delivery zone. Only call this for an item you have just confirmed "
                           "is present via perceive_scene.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "The item to deliver, in Arabic, exactly as returned by perceive_scene.",
                    }
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": "Speak a short message in Arabic to the user. This is the user's "
                           "ONLY feedback channel — always confirm what you did, or explain "
                           "what is available if you cannot fulfill the request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_ar": {
                        "type": "string",
                        "description": "The message in Arabic, matching the user's dialect. Keep it short and natural.",
                    }
                },
                "required": ["text_ar"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask",
            "description": "Ask the user a clarifying question in Arabic when the request is "
                           "AMBIGUOUS — i.e. it could refer to more than one item present on the "
                           "vanity (e.g. 'السيروم' when both a face serum and a hair serum are "
                           "present). Speak the options and wait for the user's answer; do NOT guess.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_ar": {
                        "type": "string",
                        "description": "The clarifying question in Arabic, naming the options.",
                    }
                },
                "required": ["text_ar"],
            },
        },
    },
]


def run_tool(name, args):
    """Execute a tool by name with a dict of args; always returns a dict."""
    fn = DISPATCH.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown_tool:{name}"}
    try:
        return fn(**(args or {}))
    except TypeError as e:
        return {"ok": False, "error": f"bad_args:{e}"}
