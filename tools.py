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

# What's currently on the vanity. (Arabic labels, as the user would name them.)
SCENE_ITEMS = ["عطر ديور", "كريم مرطب", "واقي شمس"]  # Dior perfume, moisturizer, sunscreen


def perceive_scene():
    """Look at the vanity and return the items currently present."""
    if os.environ.get("BASEER_VISION") == "1":
        try:
            import vision
            items = vision.perceive()
            print(f"  [tool] perceive_scene() [YOLO] -> {items}")
            return {"items": items}
        except Exception as e:
            print(f"  [tool] perceive_scene() vision failed ({e}); using stub list")
    print(f"  [tool] perceive_scene() -> {SCENE_ITEMS}")
    return {"items": list(SCENE_ITEMS)}


def deliver(item):
    """Pick `item` from the vanity and place it at the fixed delivery zone."""
    if item not in SCENE_ITEMS:
        print(f"  [tool] deliver(item='{item}') -> NOT PRESENT")
        return {"ok": False, "error": "item_not_present", "item": item,
                "available": list(SCENE_ITEMS)}
    print(f"  [tool] deliver(item='{item}') -> delivering to zone...")
    return {"ok": True, "item": item}


def say(text_ar):
    """Speak a short Arabic message to the user (their only feedback channel)."""
    print(f"  [SAY] 🔊 {text_ar}")
    return {"ok": True}


# --- dispatch + OpenAI-style tool schema -----------------------------------
DISPATCH = {"perceive_scene": perceive_scene, "deliver": deliver, "say": say}

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
