# بَصير · Baseer — an Arabic voice-controlled assistive arm

> *بَصير ≈ "the perceptive one / the one who sees."* The arm becomes the eyes and hands of a blind or hands-occupied person, driven entirely by spoken Arabic.

Baseer lets a user say, in everyday Arabic dialect — *"ناوليني العطر"* — and a robot arm checks what's actually in front of it, fetches the item, places it at a fixed delivery spot, and **speaks back in Arabic** to confirm. If the item isn't there, it doesn't guess — it says aloud what *is* available.

Built for the Fanar hackathon, **Theme 4: Physical AI / Imitation Learning**.

---

## 1. Problem statement

For a visually-impaired person, the hard part of daily life isn't *deciding* what they want — it's **locating and reaching** it. Existing robotic aids have two barriers:

1. **Control barrier** — they're joystick/teleop driven, which assumes fine motor control and sight.
2. **Language barrier** — interfaces and assistants are English-first.

For an elderly, blind, Arabic-speaking user, neither works. The accessibility gap is **natural, dialectal Arabic** as the interface to a physical assistant — and that gap is precisely what Fanar is built to close.

**Use case (demo):** the arm sits on a **vanity/dressing table** and hands the user perfume / skincare items on request. Items live in fixed, memorable positions (which is how a blind person organizes anyway). Audio is the user's **only** feedback channel.

**Value proposition:** the *intelligence* (understanding spoken dialect, deciding what to do, refusing safely) is genuinely Fanar's; the impact is real and dignified; and the design proves it by **failing gracefully** — which, for an assistive tool, *is* the product.

**Societal impact:** independence and dignity for visually-impaired Arabic speakers; a template for Arabic-first assistive robotics; and a direct fit for Fanar's sovereign-AI accessibility mission.

---

## 2. Solution architecture

Three tiers, split by a hard constraint: **the robot's USB serial and the camera are only visible to the machine they're plugged into**, so a laptop must host the control loop. The phone is a thin client (mic + speaker); the GPU box is used **offline only** to train the motion policy.

```
┌─────────────────┐    HTTPS (LAN/tunnel)    ┌──────────────────────────────────────────┐   HTTPS API   ┌──────────────────┐
│   PHONE          │ ──── audio upload ─────▶ │   LAPTOP (host / orchestrator)             │ ── requests ─▶│   FANAR CLOUD    │
│  thin client     │ ◀─── spoken reply ────── │   server.py (FastAPI)                      │ ◀ responses ──│  Aura ASR / TTS  │
│ • tap-hold UI    │                          │     ├ agent.py   (the agent loop / brain)  │               │  Fanar-C-2-27B   │
│ • records mic    │                          │     ├ tools.py   (perceive / deliver / say)│               └──────────────────┘
│ • plays MP3      │                          │     ├ vision.py  (YOLO eyes)               │   LOCAL on laptop:
└─────────────────┘                          │     └ fanar.py   (Fanar + Aura client)     │   • YOLO-World (MPS)
                                              │              │                             │   • OpenCV camera (idx 0)
                                              │              ▼                             │   • USB serial → SO-100
                                              │     SO-100 arm  (deliver = ACT / VLA)      │
                                              └──────────────────────────────────────────┘
```

**The four "organs":**

| Organ | Technology | Responsibility |
|---|---|---|
| **Brain** | Fanar-C-2-27B | understand dialect, decide *what* to do, refuse safely |
| **Eyes** | YOLO-World (local, open-vocab) | report what is actually on the vanity |
| **Voice** | Aura ASR + Aura TTS | hear the user, speak back in Arabic |
| **Muscle** | ACT / VLA on SO-100 (LeRobot) | physically grasp + deliver |

**Two pipelines:**
```
ONLINE  (per request):  voice → Aura ASR → Fanar agent loop → YOLO → deliver → Aura TTS
OFFLINE (once, on GPU):  teleop demos → record dataset → train ACT/VLA → checkpoint → plugs into deliver()
```

---

## 3. Agentic workflow design

The core is an **agentic control loop (ReAct: Reason → Act → Observe)** with **tool use**, where **Fanar is the decision-maker** and the Python is a thin executor. There is **no hardcoded `if/else` pipeline** — every semantic decision is made by Fanar, guided by a system prompt.

### Action space — three tools
```
perceive_scene()  → {"items":[...]}     # YOLO: what's on the vanity
deliver(item)     → {"ok":true/false}   # ACT/VLA: grasp + place at the fixed zone
say(text_ar)      → {"ok":true}         # Aura TTS: speak to the user
```
Fanar's entire output is *choosing among these*. That makes Fanar a **task-level VLA** (Vision–Language → Action, where actions are tool calls), with ACT/VLA as the **low-level** controller — a hierarchical VLA.

### Who decides what
| Actor | Decides | |
|---|---|---|
| **Fanar (LLM)** | which tool next, the args, when to stop, what to say, whether to refuse | *all semantics* |
| **agent.py loop** | loop ≤ MAX_STEPS, parse JSON, dispatch to the tool, retry on errors, stop on `done` | *pure mechanics* |
| **tools / vision / robot** | deterministic execution | *effectors* |

### The loop, message by message (for *"ناوليني العطر"*)
```
SYSTEM:  You are بَصير … rules: perceive before deliver; always say; fail gracefully.
USER:    ناوليني العطر
  Fanar ▶ {"action":"perceive_scene","args":{}}
OBSERVATION (perceive_scene): {"items":["عطر ديور","كريم مرطب","واقي شمس"]}
  Fanar ▶ {"action":"deliver","args":{"item":"عطر ديور"}}
OBSERVATION (deliver): {"ok":true,"item":"عطر ديور"}
  Fanar ▶ {"action":"say","args":{"text_ar":"تفضّل، العطر أمامك"}}
OBSERVATION (say): {"ok":true}
  Fanar ▶ {"action":"done","args":{}}            ← loop ends
```
The **state is the growing message list** (Fanar is stateless per call; we re-send the whole history each turn). Each iteration: send full context → Fanar returns one JSON action → server executes the real tool → appends the result as `OBSERVATION:` → repeat until `done` or `MAX_STEPS`.

### Planning & orchestration properties
- **Grounding:** Fanar must `perceive_scene` and confirm an item is present before `deliver` — it never acts on an unseen item.
- **Graceful failure:** if the requested item is absent from the observation, Fanar skips `deliver` and only `say`s what *is* available. *Same code path; the branch is decided by Fanar reading the observation.*
- **Multi-step:** "العطر والمرطب" → deliver one, then the other, then a single spoken confirmation.
- **Robustness:** content-filter / 429 / 5xx **retry with backoff**; tolerant JSON extraction (handles fences/stray text); **ASR normalization** before the loop; safe spoken fallback if no clean action.

### Why JSON-action protocol (not native function-calling)
We found Fanar-C-2-27B **accepts** the OpenAI `tools` parameter but **never emits `tool_calls`** — it just chats. So we drive it as a **JSON state machine**: `response_format={"type":"json_object"}` *forces* one JSON object per turn, and a strict few-shot prompt defines the schema and rules. This was the single change that took the agent from "narrates fake actions" to **reliable**.

---

## 4. Use of Fanar and external tools/models

### Fanar (the intelligence)
| Capability | Model ID | Role |
|---|---|---|
| **Controller / reasoning** | `Fanar-C-2-27B` | the agent brain — dialect understanding, planning, tool selection, graceful refusal |
| **Speech-to-text** | `Fanar-Aura-STT-1` | Arabic voice command → text |
| **Text-to-speech** | `Fanar-Aura-TTS-2` (voice **Noor**) | spoken Arabic replies |
| **Vision (optional)** | `Fanar-Oryx-IVU-2` | alternative `perceive_scene` that keeps perception inside Fanar |

Fanar's **dialect handling is the star**: Gulf / Egyptian / MSA and code-switching are understood natively, mapped *directly* to actions with **no English translation step** — which would only add latency and error and discard Fanar's main advantage.

### External tools/models (the body)
| Tool | Role |
|---|---|
| **YOLO-World** (`ultralytics`) | open-vocabulary perception for `perceive_scene` — zero training |
| **ACT / SmolVLA** (LeRobot) | the learned grasp-and-deliver motion (`deliver`) |
| **SO-100 arm** (Feetech STS3215) | the physical robot |
| **FastAPI + OpenCV** | host server + camera capture |
| *(planned)* local Whisper | ASR fallback for when Aura is rate-limited |

### Fanar-as-VLA (the "go deeper" angle)
Rather than fine-tuning a monolithic VLA (impossible via API — no weights, no action head, latency), Baseer realizes Fanar as a **hierarchical VLA**: Fanar = high-level Vision-Language→Action controller (actions = tool calls); ACT/VLA = low-level motor controller. The same recorded dataset can train **ACT** (reliable baseline) *and* a **language-conditioned VLA** (`smolvla`) — `--policy.type` is the only difference.

---

## 5. Evaluation results

### Functional (live, against the real Fanar API)
| Test | Input | Result |
|---|---|---|
| Happy path (MSA) | "ناولني عطر ديور" | perceive → deliver → *"تفضل، العطر أمامك"* ✅ |
| **Graceful failure** | "أبي الروج" (absent) | perceive → *"آسف، لا يوجد روج على الطاولة"* — **no delivery** ✅ |
| Egyptian dialect | "هاتلي الكريم المرطب لو سمحت" | → deliver كريم مرطب ✅ |
| Gulf dialect | "عطني واقي الشمس عساك بخير" | → deliver واقي شمس ✅ |
| Multi-item | "ناولني العطر والمرطب" | deliver عطر → deliver كريم → confirm both ✅ |

### Voice round-trip (Aura)
TTS (`Noor`) → MP3 → ASR → text: exact match (*"تفضل عطر ديور أمامك"*). Both endpoints verified.

### Engineering experiments / findings
| Finding | Evidence | Fix we built |
|---|---|---|
| Native tool-calling non-functional | `tools` accepted, `tool_calls` always `[]`, model just chats | JSON-action protocol + `response_format=json_object` + few-shot |
| **Safety filter false-positives** | benign requests intermittently → `HTTP 400 content_filter` | retry-with-backoff in the agent loop |
| **ASR mangles brand names** | "عطر ديور" → ASR "عِطراديور"; that garbled token then trips the content filter **8/8** | `normalize.py` (strip diacritics + alias-map to catalog) **before** Fanar — fixed 0/8 → working |
| Shared, low rate limits | 429 across ASR/TTS/chat under iterative testing | TTS disk-cache, fixed-phrase audio via device voice, fewer agent calls |

YOLO-World perception verified on a test image (correctly detected real objects; returned `[]` when target items absent).

---

## 6. Recommendations for future Fanar improvements

1. **Make function/tool-calling actually emit `tool_calls`.** The endpoint accepts `tools` but never returns calls, forcing every agent team into a brittle JSON-protocol workaround. Native, reliable tool-calling would be the single biggest agentic-DX win.
2. **Reduce safety-filter false positives on benign Arabic**, and/or expose the trigger + a `safe_mode`/severity setting. Today it intermittently blocks innocuous requests ("give me the perfume"), which is fatal for a deterministic agent.
3. **Aura ASR: entity/brand robustness + a diacritics toggle.** ASR fuses multi-word entities ("عطر ديور"→"عِطراديور") and returns heavy diacritics; both destabilize downstream LLM parsing. An option for plain (un-diacritized) output and better named-entity handling would help.
4. **Clearer, higher, per-capability rate limits + `Retry-After` headers.** A single low shared quota across chat/ASR/TTS makes live multimodal demos impractical; 429s carry no backoff hint.
5. **Ship a Fanar-native VLA checkpoint** (built on `Fanar-Oryx`) that's fine-tunable in the LeRobot format — this would let teams build a *genuine* Fanar VLA instead of pairing Fanar with a third-party policy.
6. **Minor API compatibility:** `GET /models` returns results under a `"models"` key instead of OpenAI's `"data"`, breaking drop-in OpenAI SDK usage. Aligning would ease adoption.

---

## 7. Setup & run

### Prerequisites
- Python env with deps: `pip install -r requirements.txt` (FastAPI, requests, python-dotenv, python-multipart, ultralytics, etc.)
- A Fanar API key.

### Configure
```bash
cp .env.example .env      # then fill in:
# FANAR_API_KEY=...        FANAR_MODEL=Fanar-C-2-27B
```

### Run (laptop)
```bash
# software demo (stub robot, real Fanar+Aura+YOLO):
python server.py
# open http://localhost:8080

# with real camera perception:
BASEER_VISION=1 python server.py

# for the phone over HTTPS (mic needs a secure context):
PORT=8443 BASEER_CERT=certs/cert.pem BASEER_KEY=certs/key.pem python server.py
# phone → https://<laptop-ip>:8443   (accept the self-signed cert)
```

### Robot (offline, see `baseer_record/`)
```bash
# record demonstrations, then train the policy on a GPU box:
./baseer_record/record_water.sh        # teleop + record
./baseer_record/train_water.sh         # lerobot-train --policy.type=act (or smolvla)
```

---

## 8. Repository structure
```
baseer/
  server.py        FastAPI host: /command-audio, /tts, /command  (+ logging, TTS cache)
  agent.py         the ReAct agent loop (JSON-action protocol, retries)
  prompts.py       system prompt + hard rules
  tools.py         action space: perceive_scene / deliver / say
  fanar.py         Fanar chat client + Aura transcribe/synthesize
  normalize.py     ASR post-correction (diacritics + entity aliases)
  vision.py        YOLO-World perception
  web/index.html   phone UI (tap-and-hold, audio in/out, accessibility)
  test_agent.py    offline agent tests (no API key needed)
  baseer_record/   robot: record + train scripts, phone record-control panel
```

## 9. Status
The full **software spine is working** on the real stack: Aura ASR → Fanar dialect reasoning → YOLO grounding → graceful failure → Aura TTS, with accessibility (VoiceOver) and rate-limit resilience. The one remaining real piece is training the motion policy so `deliver` physically grasps (`baseer_record/`).

---

*Baseer — where the intelligence is genuinely Fanar's, the impact is dignified, and the system proves it by failing gracefully.*
