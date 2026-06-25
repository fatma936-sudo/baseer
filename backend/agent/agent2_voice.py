"""
Agent 2 — Voice.  Models: Fanar-Aura-TTS-2 (speech) + Fanar-Aura-STT-1 (transcription).

The user's only feedback channel: hear the spoken Arabic command (transcribe) and
speak the reply back (synthesize).
"""
import time

import requests

from agent.fanar_base import (AURA_STT_MODEL, AURA_TTS_MODEL, AURA_VOICE,
                              FANAR_BASE_URL, FanarError, _AUTH)


def synthesize(text_ar, voice=None):
    """Aura TTS: Arabic text -> MP3 bytes. Retries on rate-limit/transient errors."""
    last = None
    for attempt in range(3):
        r = requests.post(
            f"{FANAR_BASE_URL.rstrip('/')}/audio/speech",
            headers={**_AUTH, "Content-Type": "application/json"},
            json={"model": AURA_TTS_MODEL, "input": text_ar, "voice": voice or AURA_VOICE},
            timeout=60,
        )
        if r.status_code == 200:
            return r.content
        last = f"TTS HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code in (429, 500, 502, 503):
            time.sleep(1.5 * (attempt + 1))
            continue
        break
    raise FanarError(last)


def transcribe(audio_bytes, filename="audio.webm", mime="audio/webm"):
    """Aura ASR: Arabic speech bytes -> text."""
    r = requests.post(
        f"{FANAR_BASE_URL.rstrip('/')}/audio/transcriptions",
        headers=_AUTH,
        files={"file": (filename, audio_bytes, mime)},
        data={"model": AURA_STT_MODEL},
        timeout=60,
    )
    if r.status_code != 200:
        raise FanarError(f"ASR HTTP {r.status_code}: {r.text[:300]}")
    return r.json().get("text", "")
