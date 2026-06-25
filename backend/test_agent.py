"""
Offline test of the agent loop using a FAKE Fanar client (no API key needed).
Covers: happy path, graceful failure, native-tools path, and the ask/disambiguation
multi-turn flow.

    python test_agent.py
"""
import json

from agent.orchestrator import run


def tc(name, args):
    return {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": f"call_{name}", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}],
    }


class FakeJsonClient:
    """Replays JSON-action content strings (real Fanar path)."""
    def __init__(self, contents):
        self.contents = list(contents)

    def chat(self, messages, tools=None, temperature=0.2, response_format=None):
        return {"role": "assistant", "content": self.contents.pop(0)}


class FakeToolClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages, tools=None, temperature=0.2, response_format=None):
        return self.responses.pop(0)


def delivered_ok(tr):
    return any(t["action"] == "deliver" and t["result"].get("ok") for t in tr)


results = []

print("=== 1) HAPPY · JSON mode ===")
r = run("ناولني عطر الياسمين", json_mode=True, client=FakeJsonClient([
    '{"action":"perceive_scene","args":{}}',
    '{"action":"deliver","args":{"item":"عطر الياسمين"}}',
    '{"action":"say","args":{"text_ar":"تفضّل، عطر الياسمين قدّامك"}}',
    '{"action":"done","args":{}}',
]))
ok = delivered_ok(r["transcript"]) and bool(r["reply"]) and not r["awaiting"]
print("  ->", "PASS" if ok else "FAIL"); results.append(ok)

print("=== 2) GRACEFUL FAILURE · JSON mode ===")
r = run("أبي الروج", json_mode=True, client=FakeJsonClient([
    '{"action":"perceive_scene","args":{}}',
    '{"action":"say","args":{"text_ar":"للأسف الروج غير موجود."}}',
    '{"action":"done","args":{}}',
]))
ok = (not delivered_ok(r["transcript"])) and bool(r["reply"])
print("  ->", "PASS" if ok else "FAIL"); results.append(ok)

print("=== 3) HAPPY · native tool-calling path ===")
r = run("ناولني الشامبو الجاف", json_mode=False, client=FakeToolClient([
    tc("perceive_scene", {}),
    tc("deliver", {"item": "شامبو جاف"}),
    tc("say", {"text_ar": "تفضّل، الشامبو الجاف قدّامك"}),
    {"role": "assistant", "content": ""},
]))
ok = delivered_ok(r["transcript"]) and bool(r["reply"])
print("  ->", "PASS" if ok else "FAIL"); results.append(ok)

print("=== 4) DISAMBIGUATION · ask, then continue after answer ===")
fc = FakeJsonClient([
    '{"action":"perceive_scene","args":{}}',
    '{"action":"ask","args":{"text_ar":"أي سيروم، الوجه أم الشعر؟"}}',   # turn 1 ends here
    '{"action":"deliver","args":{"item":"سيروم الوجه"}}',              # turn 2 (after answer)
    '{"action":"say","args":{"text_ar":"تفضّل، سيروم الوجه أمامك"}}',
    '{"action":"done","args":{}}',
])
r1 = run("ناولني السيروم", json_mode=True, client=fc)
asked = any(t["action"] == "ask" for t in r1["transcript"])
r2 = run("سيروم الوجه", json_mode=True, client=fc, history=r1["messages"])
ok = r1["awaiting"] and asked and delivered_ok(r2["transcript"])
print(f"  -> {'PASS' if ok else 'FAIL'} (turn1 asked={asked} awaiting={r1['awaiting']}, turn2 delivered={delivered_ok(r2['transcript'])})")
results.append(ok)

print("=" * 44)
print("ALL PASSED ✅" if all(results) else "SOME FAILED ❌")
