"""
Offline test of the agent loop using a FAKE Fanar client (no API key needed).
Tests the JSON-action path (what real Fanar uses) plus the tool-calling path.

    python test_agent.py
"""
import json

from agent import run


def tc(name, args):
    """Build a fake tool-calling assistant message."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        }],
    }


class FakeJsonClient:
    """Replays JSON-action content strings (real Fanar path)."""
    def __init__(self, contents):
        self.contents = list(contents)

    def chat(self, messages, tools=None, temperature=0.2, response_format=None):
        return {"role": "assistant", "content": self.contents.pop(0)}


class FakeToolClient:
    """Replays tool-calling responses (for the dormant native-tools path)."""
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages, tools=None, temperature=0.2, response_format=None):
        return self.responses.pop(0)


def check(transcript, expect_deliver):
    delivered = [t for t in transcript if t["action"] == "deliver" and t["result"].get("ok")]
    said = [t for t in transcript if t["action"] == "say"]
    ok = bool(delivered) == expect_deliver and bool(said)
    print(f"  -> {'PASS' if ok else 'FAIL'} (delivered={bool(delivered)}, spoke={bool(said)})\n")
    return ok


results = []

print("=== 1) HAPPY PATH · JSON mode (real Fanar path) ===")
t = run("ناولني العطر", json_mode=True, client=FakeJsonClient([
    '{"action":"perceive_scene","args":{}}',
    '{"action":"deliver","args":{"item":"عطر ديور"}}',
    '{"action":"say","args":{"text_ar":"تفضّل، العطر قدّامك"}}',
    '{"action":"done","args":{}}',
]))
results.append(check(t, expect_deliver=True))

print("=== 2) GRACEFUL FAILURE · JSON mode (lipstick not on vanity) ===")
t = run("أبي الروج", json_mode=True, client=FakeJsonClient([
    '{"action":"perceive_scene","args":{}}',
    '{"action":"say","args":{"text_ar":"للأسف الروج غير موجود. المتوفّر: عطر ديور، كريم مرطب، واقي شمس."}}',
    '{"action":"done","args":{}}',
]))
results.append(check(t, expect_deliver=False))

print("=== 3) HAPPY PATH · native tool-calling path (forced) ===")
t = run("ناولني الكريم", json_mode=False, client=FakeToolClient([
    tc("perceive_scene", {}),
    tc("deliver", {"item": "كريم مرطب"}),
    tc("say", {"text_ar": "تفضّل، الكريم قدّامك"}),
    {"role": "assistant", "content": ""},
]))
results.append(check(t, expect_deliver=True))

print("=" * 44)
print("ALL PASSED ✅" if all(results) else "SOME FAILED ❌")
