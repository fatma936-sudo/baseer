"""
Baseer agent loop: perceive -> decide -> act -> observe.

Supports BOTH native tool-calling and a JSON-action fallback. It tries
tool-calling first; if the model/tier rejects tools, it transparently
re-runs the turn in JSON-action mode.

Run it:
    export FANAR_API_KEY=...        # and FANAR_MODEL if needed
    python agent.py "ناولني القهوة"
"""
import json

from fanar import FanarClient, FanarError
from tools import TOOLS, run_tool
from prompts import build_system_prompt

MAX_STEPS = 10

# Fanar accepts the `tools` param but does not actually emit tool_calls, so we
# drive it via the JSON-action protocol + response_format=json_object instead.
USE_NATIVE_TOOLS = False


def _is_tool_unsupported(err):
    s = str(err).lower()
    return "tool" in s or "function" in s or "http 400" in s or "http 422" in s


# Fanar's safety filter fires intermittently (HTTP 400 content_filter); a re-roll
# almost always passes. Also retry transient 429/5xx.
_RETRYABLE = ("content_filter", "safety", "http 429", "http 500", "http 502", "http 503")


def _chat(client, messages, tools, response_format, tries=4):
    last = None
    for i in range(tries):
        try:
            # vary temperature across retries to dodge the intermittent safety filter
            return client.chat(messages, tools=tools, temperature=0.2 + 0.25 * i,
                               response_format=response_format)
        except FanarError as e:
            last = e
            if any(k in str(e).lower() for k in _RETRYABLE):
                continue
            raise
    raise last


def _parse_json_obj(text):
    """Pull the first {...} JSON object out of a model message, tolerating fences."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None


def _extract_actions(msg, json_mode):
    """Normalize a model message into a list of {name, args, id} actions."""
    if json_mode:
        obj = _parse_json_obj(msg.get("content", "")) or {}
        action = obj.get("action")
        if not action:
            return []
        if action in ("done", "finish", "stop"):
            return [{"name": "done", "args": {}, "id": None}]
        return [{"name": action, "args": obj.get("args", {}) or {}, "id": None}]

    actions = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        actions.append({"name": fn.get("name"), "args": args, "id": tc.get("id")})
    return actions


def _assistant_msg(msg, json_mode):
    """The assistant turn to append back into the conversation."""
    if json_mode:
        return {"role": "assistant", "content": msg.get("content", "")}
    return msg  # keep tool_calls intact for the API


def _observation_msg(name, result, call_id, json_mode):
    payload = json.dumps(result, ensure_ascii=False)
    if json_mode:
        return {"role": "user", "content": f"OBSERVATION ({name}): {payload}"}
    return {"role": "tool", "tool_call_id": call_id, "content": payload}


def _loop(client, user_text, json_mode, verbose, history=None):
    if history:                                   # continue an existing conversation
        messages = list(history) + [{"role": "user", "content": user_text}]
    else:
        messages = [
            {"role": "system", "content": build_system_prompt(json_mode=json_mode)},
            {"role": "user", "content": user_text},
        ]
    tools = None if json_mode else TOOLS
    response_format = {"type": "json_object"} if json_mode else None
    transcript = []
    awaiting = False

    for _ in range(MAX_STEPS):
        msg = _chat(client, messages, tools, response_format)
        actions = _extract_actions(msg, json_mode)

        if not actions:
            final = (msg.get("content") or "").strip()
            if verbose and final:
                print(f"  [model] {final}")
            break

        messages.append(_assistant_msg(msg, json_mode))

        stop = False
        for act in actions:
            if act["name"] == "done":
                stop = True
                continue
            result = run_tool(act["name"], act["args"])
            transcript.append({"action": act["name"], "args": act["args"], "result": result})
            messages.append(_observation_msg(act["name"], result, act["id"], json_mode))
            if act["name"] == "ask":              # asked a question -> stop & wait for the user
                awaiting = True
                stop = True
                break
        if stop:
            break

    reply = " ".join(t["args"].get("text_ar", "")
                     for t in transcript if t["action"] in ("say", "ask"))
    return {"transcript": transcript, "reply": reply, "awaiting": awaiting, "messages": messages}


def run(user_text, client=None, verbose=True, json_mode=None, history=None):
    """Run one user turn. Returns {transcript, reply, awaiting, messages}.

    Pass `history` (the prior `messages`) to continue a multi-turn conversation,
    e.g. after the agent asked a clarifying question.
    """
    client = client or FanarClient()
    if verbose:
        print(f"  [user] {user_text}")
    if json_mode is None:
        json_mode = not USE_NATIVE_TOOLS
    if not json_mode:
        try:
            return _loop(client, user_text, json_mode=False, verbose=verbose, history=history)
        except FanarError as e:
            if _is_tool_unsupported(e):
                if verbose:
                    print("  [agent] native tool-calling unsupported -> JSON-action fallback")
                return _loop(client, user_text, json_mode=True, verbose=verbose, history=history)
            raise
    return _loop(client, user_text, json_mode=True, verbose=verbose, history=history)


if __name__ == "__main__":
    import sys
    request = " ".join(sys.argv[1:]) or "ناولني القهوة"
    run(request)
