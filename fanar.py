"""
Thin Fanar API client (OpenAI-compatible chat completions).

Config via environment (or a .env file in this folder):
    FANAR_API_KEY   - your key                 (required)
    FANAR_BASE_URL  - default https://api.fanar.qa/v1
    FANAR_MODEL     - the chat model id        (confirm in your Fanar dashboard)

Aura ASR/TTS are stubbed at the bottom — fill the endpoint paths once you
confirm them in the dashboard. The agent-on-stubs milestone doesn't need them.
"""
import os
import json
import requests

# Optional: load a local .env if python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

FANAR_BASE_URL = os.environ.get("FANAR_BASE_URL", "https://api.fanar.qa/v1")
FANAR_API_KEY = os.environ.get("FANAR_API_KEY", "")
# CONFIRM this in your dashboard. Per your doc the controller is Fanar-2-27B-Instruct;
# the public general model is often just "Fanar". Set FANAR_MODEL to whatever you have.
FANAR_MODEL = os.environ.get("FANAR_MODEL", "Fanar")


class FanarError(RuntimeError):
    pass


class FanarClient:
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key if api_key is not None else FANAR_API_KEY
        self.base_url = (base_url or FANAR_BASE_URL).rstrip("/")
        self.model = model or FANAR_MODEL
        if not self.api_key:
            raise FanarError("FANAR_API_KEY is not set. Run: export FANAR_API_KEY=...")

    def chat(self, messages, tools=None, temperature=0.2, response_format=None):
        """Return the assistant message dict (may contain 'tool_calls')."""
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        if resp.status_code != 200:
            raise FanarError(f"HTTP {resp.status_code}: {resp.text[:600]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError):
            raise FanarError(f"Unexpected response shape: {json.dumps(data)[:600]}")


# ---------------------------------------------------------------------------
# Aura ASR / TTS (verified live).
# ---------------------------------------------------------------------------
AURA_TTS_MODEL = os.environ.get("FANAR_TTS_MODEL", "Fanar-Aura-TTS-2")
AURA_STT_MODEL = os.environ.get("FANAR_STT_MODEL", "Fanar-Aura-STT-1")
AURA_VOICE = os.environ.get("FANAR_VOICE", "Noor")  # Noor/Huda/Radwa (F), Jasim/Hamad/Abdulrahman (M)

_AUTH = {"Authorization": f"Bearer {FANAR_API_KEY}"}


def synthesize(text_ar, voice=None):
    """Aura TTS: Arabic text -> MP3 bytes. Retries on rate-limit/transient errors."""
    import time
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
