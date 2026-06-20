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

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, Response

from agent import run as agent_run
from fanar import FanarClient, FanarError, synthesize, transcribe
from normalize import normalize_command
import tools as T

ACK = "تم استلام الأمر، قيد التنفيذ"

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
WEB = Path(__file__).parent / "web"


# --- DEV MOCK (used only when FANAR_API_KEY is not set) ---------------------
# Lets you click through the whole UI + agent loop before wiring real Fanar.
_SYNONYMS = {
    "عطر ديور": ["عطر", "ديور", "برفان", "بارفان", "perfume", "dior"],
    "كريم مرطب": ["مرطب", "مرطّب", "كريم", "moistur", "cream"],
    "واقي شمس": ["واقي", "شمس", "صن", "sunscreen", "spf"],
}


def _resolve_item(text):
    for item, kws in _SYNONYMS.items():
        if item in T.SCENE_ITEMS and any(k in text for k in kws):
            return item
    return None


class MockClient:
    model = "mock-dev"

    def chat(self, messages, tools=None, temperature=0.2, response_format=None):
        user = next(
            (m["content"] for m in messages
             if m["role"] == "user" and not str(m["content"]).startswith("OBSERVATION")),
            "",
        )
        issued = [tc["function"]["name"]
                  for m in messages if m.get("tool_calls")
                  for tc in m["tool_calls"]]

        def call(name, args):
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{name}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }]}

        if "perceive_scene" not in issued:
            return call("perceive_scene", {})
        item = _resolve_item(user)
        if item and "deliver" not in issued:
            return call("deliver", {"item": item})
        if "say" not in issued:
            if item:
                txt = f"تفضّل، {item} قدّامك."
            else:
                txt = "للأسف هذا غير موجود على التسريحة. المتوفّر: " + "، ".join(T.SCENE_ITEMS) + "."
            return call("say", {"text_ar": txt})
        return {"role": "assistant", "content": ""}


def get_client():
    return FanarClient() if os.environ.get("FANAR_API_KEY") else MockClient()


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
        transcript = agent_run(text, client=get_client(), verbose=False)
    except FanarError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    reply = " ".join(t["args"].get("text_ar", "")
                     for t in transcript if t["action"] == "say") or "تم."
    return {"heard": text, "reply": reply, "actions": transcript,
            "engine": get_client().model}


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
async def command_audio(file: UploadFile = File(...)):
    """Voice command: audio -> Aura ASR -> agent -> {heard, reply}."""
    audio = await file.read()
    try:
        raw = transcribe(audio, filename=file.filename or "audio.webm",
                         mime=file.content_type or "audio/webm").strip()
    except FanarError as e:
        print(f"[asr] FAILED: {e}")
        return JSONResponse({"error": f"asr: {e}"}, status_code=502)
    text = normalize_command(raw)
    if not text:
        log_turn(audio, raw, text, "")
        return JSONResponse({"error": "no speech detected", "raw_asr": raw}, status_code=400)
    try:
        transcript = agent_run(text, client=get_client(), verbose=False)
    except FanarError as e:
        print(f"[agent] FAILED: {e}")
        log_turn(audio, raw, text, f"ERROR: {e}")
        return JSONResponse({"heard": text, "error": str(e)}, status_code=502)
    reply = " ".join(t["args"].get("text_ar", "")
                     for t in transcript if t["action"] == "say") or "تم."
    log_turn(audio, raw, text, reply)
    return {"heard": text, "raw_asr": raw, "reply": reply, "actions": transcript}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    cert, key = os.environ.get("BASEER_CERT"), os.environ.get("BASEER_KEY")
    if cert and key:
        print(f"HTTPS on https://0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, ssl_certfile=cert, ssl_keyfile=key)
    else:
        uvicorn.run(app, host="0.0.0.0", port=port)
