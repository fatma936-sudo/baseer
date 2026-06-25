"""
Shared Fanar transport: base URL, API key, model ids, auth header, and the common
error type. Each per-model "agent" module (agent1 reasoning, agent2 voice, agent3
vision) imports its config from here so there's one place for credentials + model ids.

Config via environment (or a .env file at the repo root / backend/):
    FANAR_API_KEY   - your key                       (required for live calls)
    FANAR_BASE_URL  - default https://api.fanar.qa/v1
    FANAR_MODEL     - the chat model id (e.g. Fanar-C-2-27B)
"""
import os

# Load a local .env if python-dotenv is available — try repo root and backend/.
try:
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__))            # backend/agent
    _repo_root = os.path.dirname(os.path.dirname(_here))          # repo root
    for _cand in (os.path.join(_repo_root, ".env"),
                  os.path.join(os.path.dirname(_here), ".env")):  # backend/.env
        if os.path.exists(_cand):
            load_dotenv(_cand)
            break
except Exception:
    pass

FANAR_BASE_URL = os.environ.get("FANAR_BASE_URL", "https://api.fanar.qa/v1")
FANAR_API_KEY = os.environ.get("FANAR_API_KEY", "")
FANAR_MODEL = os.environ.get("FANAR_MODEL", "Fanar")

# Aura (voice) + Oryx (vision) model ids.
AURA_TTS_MODEL = os.environ.get("FANAR_TTS_MODEL", "Fanar-Aura-TTS-2")
AURA_STT_MODEL = os.environ.get("FANAR_STT_MODEL", "Fanar-Aura-STT-1")
AURA_VOICE = os.environ.get("FANAR_VOICE", "Noor")  # Noor/Huda/Radwa (F), Jasim/Hamad/Abdulrahman (M)
VISION_MODEL = os.environ.get("FANAR_VISION_MODEL", "Fanar-Oryx-IVU-2")

_AUTH = {"Authorization": f"Bearer {FANAR_API_KEY}"}


class FanarError(RuntimeError):
    pass
