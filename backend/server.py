"""
Baseer backend: serves the tap-to-talk web app and runs the Fanar agent.

    cd ~/baseer
    export FANAR_API_KEY=...        # optional; without it a DEV MOCK is used
    python server.py
    # open http://localhost:8080  (laptop)

Endpoints:
    GET  /          -> the web app (white screen, tap to talk)
    POST /command   -> {"text": "..."} -> runs the agent -> {"heard","reply","actions"}

Note on phone access: browser mic capture needs a SECURE context (https or
localhost). Over plain http://<lan-ip> the phone will block the mic. For the
phone demo, expose it via a tunnel (e.g. `cloudflared tunnel --url
http://localhost:8080`) which gives an https URL.
"""
import os
import json
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, Response

from agent.orchestrator import run as agent_run
from agent.agent1_reasoning import FanarClient, FanarError
from agent.agent2_voice import synthesize, transcribe
from normalize import normalize_command
import tools as T

ACK = "تم استلام الأمر، قيد التنفيذ"
FALLBACK = "عذراً، ما قدرت أُكمل طلبك، حاول مرة ثانية"   # spoken on any agent failure (never raw errors)
NOHEAR = "عذراً، ما سمعتك بوضوح، حاول مرة ثانية"

import time
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(os.path.join(LOG_DIR, "audio"), exist_ok=True)


def log_turn(audio, raw, normalized, reply):
    """Append every voice turn (raw ASR, normalized, reply) to logs/transcripts.jsonl
    and save the audio, so you can verify what Aura actually heard."""
    ts = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    apath = os.path.join(LOG_DIR, "audio", ts + ".webm")
    try:
        with open(apath, "wb") as f:
            f.write(audio)
    except Exception:
        apath = None
    rec = {"ts": ts, "raw_asr": raw, "normalized": normalized, "reply": reply, "audio": apath}
    with open(os.path.join(LOG_DIR, "transcripts.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[log] raw={raw!r} -> norm={normalized!r} | reply={reply!r}")

app = FastAPI()
WEB = Path(__file__).parent.parent / "GUI"   # repo_root/GUI


# --- DEV MOCK (used only when FANAR_API_KEY is not set) ---------------------
# Lets you click through the whole UI + agent loop before wiring real Fanar.
_SYNONYMS = {
    "عطر شيا": ["شيا", "shea"],
    "عطر المسك الأبيض": ["مسك", "المسك", "musk"],
    "عطر الياسمين": ["ياسمين", "الياسمين", "jasmine"],
    "شامبو جاف": ["شامبو", "dry shampoo"],
    "بودرة الشعر": ["بودرة", "باودر", "powder"],
    "سيروم الشعر": ["كيراستاز", "hair serum"],
    "سيروم الوجه": ["فيشي", "نورمادرم", "face serum", "night"],
}


def _resolve_item(text):
    for item, kws in _SYNONYMS.items():
        if item in T.SCENE_ITEMS and any(k in text for k in kws):
            return item
    return None


class MockClient:
    """Dev brain (no key): emits JSON actions, handles serum ambiguity with ask."""
    model = "mock-dev"

    def chat(self, messages, tools=None, temperature=0.2, response_format=None):
        user_all = " ".join(
            str(m.get("content", "")) for m in messages
            if m["role"] == "user" and not str(m.get("content", "")).startswith("OBSERVATION")
        )
        issued = set()
        for m in messages:
            if m["role"] == "assistant":
                try:
                    issued.add(json.loads(m["content"]).get("action"))
                except Exception:
                    pass

        def act(a, args=None):
            return {"role": "assistant",
                    "content": json.dumps({"action": a, "args": args or {}}, ensure_ascii=False)}

        if "perceive_scene" not in issued:
            return act("perceive_scene")
        item = _resolve_item(user_all)
        if item is None and "سيروم" in user_all and "ask" not in issued:
            return act("ask", {"text_ar": "أي سيروم تريد، سيروم الوجه أم سيروم الشعر؟"})
        if item and "deliver" not in issued:
            return act("deliver", {"item": item})
        if "say" not in issued:
            txt = (f"تفضّل، {item} قدّامك." if item
                   else "للأسف غير متوفّر. المتوفّر: " + "، ".join(T.SCENE_ITEMS))
            return act("say", {"text_ar": txt})
        return act("done")


def get_client():
    return FanarClient() if os.environ.get("FANAR_API_KEY") else MockClient()


# --- multi-turn sessions (for clarifying questions) ------------------------
SESSIONS = {}  # session_id -> conversation messages, kept only while awaiting a reply


def _next_sid():
    import uuid
    return uuid.uuid4().hex[:12]


def _run_turn(text, session_id):
    """Run one user turn, continuing a session if one is awaiting an answer."""
    history = SESSIONS.pop(session_id, None) if session_id else None
    result = agent_run(text, client=get_client(), verbose=False, history=history)
    sid = session_id or _next_sid()
    if result["awaiting"]:
        SESSIONS[sid] = result["messages"]
    else:
        SESSIONS.pop(sid, None)
    return sid, (result["reply"] or "تم."), result


# --- routes ----------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.post("/command")
async def command(req: Request):
    body = await req.json()
    text = normalize_command((body.get("text") or "").strip())
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    try:
        sid, reply, result = _run_turn(text, body.get("session_id", ""))
    except FanarError as e:
        print(f"[agent] FAILED: {e}")
        return {"heard": text, "reply": FALLBACK, "awaiting": False, "session_id": body.get("session_id", "")}
    return {"heard": text, "reply": reply, "awaiting": result["awaiting"],
            "session_id": sid, "actions": result["transcript"]}


import hashlib
TTS_CACHE = os.path.join(os.path.dirname(__file__), "tts_cache")
os.makedirs(TTS_CACHE, exist_ok=True)


@app.get("/tts")
def tts(text: str, voice: str = ""):
    """Arabic text -> MP3 audio (cached on disk; fixed phrases hit Aura once)."""
    key = hashlib.md5((voice + "|" + text).encode("utf-8")).hexdigest()
    path = os.path.join(TTS_CACHE, key + ".mp3")
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/mpeg")
    try:
        audio = synthesize(text, voice=voice or None)
    except FanarError as e:
        print(f"[tts] FAILED: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)
    with open(path, "wb") as f:
        f.write(audio)
    return Response(content=audio, media_type="audio/mpeg")


@app.post("/command-audio")
async def command_audio(file: UploadFile = File(...), session_id: str = Form("")):
    """Voice command: audio -> Aura ASR -> agent -> {heard, reply, awaiting, session_id}."""
    audio = await file.read()
    try:
        raw = transcribe(audio, filename=file.filename or "audio.webm",
                         mime=file.content_type or "audio/webm").strip()
    except FanarError as e:
        print(f"[asr] FAILED: {e}")
        return {"heard": "", "reply": NOHEAR, "awaiting": False, "session_id": session_id or ""}
    text = normalize_command(raw)
    if not text:
        log_turn(audio, raw, text, "")
        return {"heard": "", "reply": NOHEAR, "awaiting": False, "session_id": session_id or ""}
    try:
        sid, reply, result = _run_turn(text, session_id)
    except FanarError as e:
        print(f"[agent] FAILED: {e}")          # logged for us; user hears a clean retry
        log_turn(audio, raw, text, f"ERROR: {e}")
        SESSIONS.pop(session_id, None)          # reset the dialog on failure
        return {"heard": text, "reply": FALLBACK, "awaiting": False, "session_id": session_id or ""}
    log_turn(audio, raw, text, reply)
    return {"heard": text, "raw_asr": raw, "reply": reply,
            "awaiting": result["awaiting"], "session_id": sid}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    cert, key = os.environ.get("BASEER_CERT"), os.environ.get("BASEER_KEY")
    if cert and key:
        print(f"HTTPS on https://0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=cert, ssl_keyfile=key)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)
