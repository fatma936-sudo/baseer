# بَصير · Baseer — an Arabic voice-controlled assistive arm

> *بَصير ≈ "the perceptive one / the one who sees."* The arm becomes the eyes and hands of a blind or hands-occupied person, driven entirely by spoken Arabic.

Baseer lets a user say in everyday Arabic dialect — *"ناوليني سيروم الشعر"* — and a robot arm **sees** what's actually on the table, **localizes** the item, **grasps** it with a learned policy, **retries if it misses**, **delivers** it to a fixed hand-off zone, and **speaks back in Arabic** to confirm. If the item isn't there, it doesn't guess — it says aloud what *is* available.

Built for the Fanar hackathon, **Theme 4: Physical AI / Imitation Learning**.

| | |
|---|---|
| 🤖 **Model weights** (SmolVLA) | https://huggingface.co/55CancriE/baseer-smolvla-serums |
| 📦 **Dataset** (LeRobot, 28 episodes) | https://huggingface.co/datasets/55CancriE/baseer_serums |
| 🌐 **Project page** | https://fatma936-sudo.github.io/baseer |
| 🦾 **Robot** | SO-100 (Feetech STS3215), single front camera |

---

## 1. Problem statement

For a visually-impaired person, the hard part of daily life isn't *deciding* what they want — it's **locating and reaching** it. Existing robotic aids have two barriers:

1. **Control barrier** — they're joystick/teleop driven, which assumes fine motor control and sight.
2. **Language barrier** — interfaces and assistants are English-first.

For an elderly, blind, Arabic-speaking user, neither works. The accessibility gap is **natural, dialectal Arabic** as the interface to a physical assistant — exactly what Fanar is built to close.

**Use case (demo):** the arm sits on a **vanity / dressing table** and hands the user **skincare serums** (hair serum, face serum) on request. Audio is the user's **only** feedback channel. The intelligence (understanding dialect, deciding, refusing safely) is  Fanar's. As Fanar-C-2-27B acts as the full ReACT agent and is the system orchestrator. 

---

## 2. System architecture

Three tiers, split by a hard constraint: **the robot's USB serial and the camera are only visible to the machine they're plugged into**, so a laptop hosts the control loop. The phone is a thin client (mic + speaker); the GPU box is used **offline only** to train the motion policy.

```
┌─────────────┐  HTTPS   ┌──────────────────────────────────────────────┐  HTTPS API  ┌────────────────┐
│   PHONE     │ ─audio─▶ │   LAPTOP  (host / orchestrator)                │ ─requests─▶ │  FANAR CLOUD   │
│ thin client │ ◀reply── │   server.py (FastAPI)                          │ ◀responses─ │  Aura ASR/TTS  │
│ tap-to-talk │          │     agent.py   ReAct loop (the brain)          │             │  Fanar-C-2-27B │
└─────────────┘          │     tools.py   perceive / deliver / say / ask  │             │  Fanar-Oryx    │
                         │     fanar.py   Fanar + Aura + Oryx client      │             └────────────────┘
                         │     grasp.py   localize → grasp → retry → give │  LOCAL on laptop:
                         │        │                                       │   • OpenCV camera (front)
                         │        ▼                                       │   • USB serial → SO-100
                         │   SO-100 arm  +  SmolVLA grasp policy          │
                         └──────────────────────────────────────────────┘
            OFFLINE (once, GPU box):  teleop demos → lerobot-record → lerobot-train (SmolVLA) → checkpoint
```

**The four "organs":**

|  Model | Task|
|---|---|
|`Fanar-C-2-27B` | understand dialect, decide *what* to do, disambiguate, refuse safely |
|`Fanar-Oryx-IVU-2` (VLM) | read labels, identify *which* serum, localize *where* it is |
| Aura ASR + Aura TTS (voice **Noor**) | hear the user, speak back in Arabic |
| **SmolVLA** on SO-100 (LeRobot) | physically grasp + deliver, with closed-loop retry |

---

## 3. The full pipeline

### Online — one spoken request
```
voice → Aura ASR → normalize → Fanar agent loop ─▶ perceive_scene (Oryx)
                                                 ─▶ deliver(item)  ────────┐
                                                 ─▶ say / ask (Aura TTS)   │
                                                                           ▼
                                            ┌──────────────  deliver(item) = grasp.py ──────────────┐
                                            │ 1. Oryx LOCATES the item → pixel (u,v)                 │
                                            │ 2. localization map: pixel → arm HOVER pose above it   │
                                            │ 3. SmolVLA policy: final descent + grasp              │
                                            │ 4. VERIFY grasp: torque (current) + gripper width      │
                                            │ 5. miss? → open, re-home, re-localize, RETRY (×N)      │
                                            │ 6. held? → scripted DELIVERY to the fixed zone → release│
                                            └────────────────────────────────────────────────────────┘
```

### Offline — train the muscle (once)
```
teleop demos → lerobot-record  →  LeRobot dataset (28 eps)  →  lerobot-train --policy.type=smolvla  →  checkpoint
```

### Why a *hybrid* (perception-guided) grasp
A pure imitation policy trained on a modest dataset learns the grasp *motion* but localizes imprecisely. Baseer keeps the imitation-learning component (the theme) **and** fixes targeting by letting **Oryx do coarse localization** and the **policy do the fine grasp**. This offloads the hard part (where is it?) from the thin-data policy onto the perception model — the same Fanar-native "eyes" that already identify the item.

### Grasp verification & retry — a "smart" grasp
Two sensor-free signals from the Feetech gripper motor decide success, with thresholds learned by `calibrate_grasp.py`:
- **Torque** — `Present_Current`: pressing on an object keeps current high; an empty closed gripper settles low.
- **Width** — `Present_Position`: an empty gripper closes all the way; an object holds the fingers open at its width.

On a miss the controller opens, returns to a known pose, re-localizes and tries again — instead of blindly continuing with an empty claw. *(True recovery skill also comes from data: include a few "miss-then-reapproach" demos.)*

---

## 4. Agentic workflow (Fanar as a hierarchical VLA)

The core is a **ReAct loop (Reason → Act → Observe)** with **tool use**, where **Fanar is the decision-maker** and Python is a thin executor — **no hardcoded `if/else` pipeline**.

### Action space — four tools
```
perceive_scene()  → {"items":[...]}     # Oryx: what's on the table
deliver(item)     → {"ok":true/false}   # localize → SmolVLA grasp → retry → deliver
say(text_ar)      → {"ok":true}         # Aura TTS: speak to the user
ask(text_ar)      → {"awaiting":true}   # clarify when a request is ambiguous (two serums)
```
Fanar's output is *choosing among these* → Fanar is a **task-level VLA** (Vision-Language → tool actions); SmolVLA is the **low-level** motor controller. A **hierarchical VLA**.

### The loop, message by message (for *"ناوليني السيروم"* with two serums present)
```
SYSTEM:  You are بَصير … perceive before deliver; always say; disambiguate; fail gracefully.
USER:    اديني السيروم
  Fanar ▶ {"action":"perceive_scene","args":{}}
OBSERVATION: {"items":["سيروم الشعر","سيروم الوجه"]}
  Fanar ▶ {"action":"ask","args":{"text_ar":"عندي سيروم الوجه وسيروم الشعر، أي واحد  تريد؟"}}   ← waits
USER:    سيروم الشعر
  Fanar ▶ {"action":"deliver","args":{"item":"سيروم الشعر"}}
OBSERVATION: {"ok":true,"item":"سيروم الشعر"}
  Fanar ▶ {"action":"say","args":{"text_ar":"تفضّل، سيروم الشعر أمامك"}}
  Fanar ▶ {"action":"done"}
```
State is the growing message list (Fanar is stateless per call; we re-send history each turn). Multi-turn disambiguation is held in server `SESSIONS`.

### Why a JSON-action protocol (not native function-calling)
Fanar-C-2-27B **accepts** the OpenAI `tools` parameter but **never emits `tool_calls`** — it just chats. So we drive it as a **JSON state machine**: `response_format={"type":"json_object"}` forces one JSON action per turn, with a strict few-shot prompt. This was the single change that made the agent reliable.

---

## 5. Dataset & model

### Dataset — `55CancriE/baseer_serums`
- **28 episodes** = 14 hair serum + 14 face serum, single front camera (640×480 @ 30 fps), LeRobot format.
- Two language tasks: *"Pick up the hair serum / face serum and place it in the delivery zone."*
- Objects placed at varied positions (free placement) so the policy generalizes across the table.
- Quality-checked: firm gripper closes on every episode (travel 68–100), no calibration/wrist anomalies.

### Model — `55CancriE/baseer-smolvla-serums`
- **SmolVLA** (SmolVLM2-500M backbone, ~450M params, ~100M trainable action expert), language-conditioned.
- Trained 20 000 steps, batch 32, **final loss 0.012** on an RTX 6000 (~6.5 h).
- Built from dataset features (`--policy.type=smolvla`) → single-camera config; deploys on Apple Silicon (MPS) or CUDA.
- Drop-in swappable with ACT / π0 / GR00T (same LeRobot dataset + the policy-agnostic loader in `grasp.py`).

---

## 6. Use of Fanar

| Capability | Model ID | Role |
|---|---|---|
| **Controller / reasoning** | `Fanar-C-2-27B` | agent brain — dialect understanding, planning, tool selection, refusal |
| **Speech-to-text** | `Fanar-Aura-STT-1` | Arabic voice command → text |
| **Text-to-speech** | `Fanar-Aura-TTS-2` (Noor) | spoken Arabic replies |
| **Vision** | `Fanar-Oryx-IVU-2` | reads labels → identifies *which* serum and localizes *where* |

Fanar's **dialect handling is the star**: Gulf / Egyptian / MSA and code-switching map *directly* to actions, no English translation step. Oryx keeps perception Fanar-native — it reads printed brand labels to tell visually-similar products apart (which plain object detectors cannot) and supplies the localization pixel that guides the arm.

---

## 7. Results & findings

### Functional (live, real Fanar API)
| Test | Input | Result |
|---|---|---|
| Happy path | "ناولني سيروم الشعر" | perceive → localize → grasp → deliver → *"تفضّل، سيروم الشعر أمامك"* ✅ |
| **Disambiguation** | "ناولني السيروم" (two present) | asks *"أي واحد تبي؟"*, waits, then delivers the chosen one ✅ |
| **Graceful failure** | absent item | perceive → speaks what *is* available, **no delivery** ✅ |
| Dialects | Egyptian / Gulf / MSA | understood natively → correct action ✅ |
| Voice round-trip | TTS→MP3→ASR | exact text match ✅ |

### Robot (physical)
- SmolVLA trained to **loss 0.012**; executes the correct grasp motion at ~20 Hz on MPS.
- Closed-loop **grasp verification** (torque + width) + **retry on miss** working.
- **Oryx-guided localization** pre-positions the arm above the object before the policy's final grasp.

### Engineering findings (and the fixes we built)
| Finding | Fix |
|---|---|
| Native tool-calling non-functional (`tool_calls` always `[]`) | JSON-action protocol + `response_format=json_object` + few-shot |
| Safety-filter false-positives on benign Arabic | retry-with-backoff in the agent loop |
| ASR mangles brand names / diacritics → trips filter | `normalize.py` (strip diacritics + alias-map) **before** Fanar |
| YOLO-World can't tell look-alike cosmetics apart | switched perception to **Fanar-Oryx** (reads labels) |
| SmolVLA feature-mismatch (3 cams vs 1) | train with `--policy.type=smolvla` (build config from dataset) |
| Deploy on Mac: processors saved with `cuda` | override processor device to `mps`/`cpu` at load |
| Policy localizes imprecisely on thin data | **Oryx-guided pre-positioning** (hybrid grasp) |

---

## 8. Setup & run

### Install
```bash
pip install -r requirements.txt
cp .env.example .env     # fill FANAR_API_KEY=...  FANAR_MODEL=Fanar-C-2-27B
```

### Voice / agent demo (laptop)
```bash
cd backend
python server.py                      # software demo (stub robot, real Fanar+Aura+Oryx)
# open http://localhost:8080
BASEER_PERCEIVE=oryx python server.py # real camera perception via Fanar-Oryx
python test_agent.py                  # offline agent tests (no API key needed)
```

### Robot — train the policy (offline, GPU box)
```bash
# 1. record demos (see baseer_record/RECORDING_GUIDE.md)
lerobot-record --robot.type=so100_follower ... --dataset.single_task="Pick up the hair serum ..."
# 2. train SmolVLA
lerobot-train --dataset.root=<dataset> --policy.type=smolvla --policy.device=cuda \
              --batch_size=32 --steps=20000 --output_dir=outputs/train/baseer_serums
```

### Robot — deploy the grasp (laptop with the arm)
```bash
# one-time calibration (arm only):
python backend/robot/calibrate_grasp.py        --port <follower> --id follower_so100  # grasp_cfg.json
python backend/robot/save_delivery_pose.py     --port <follower> --id follower_so100  # delivery_pose.json
python backend/robot/calibrate_localization.py --port <follower> --id follower_so100  # localization_map.json

# run the full localize → grasp → retry → deliver (agent 4 = SmolVLA):
python backend/agent/agent4_grasp.py --policy ~/baseer/policy_vla/pretrained_model \
  --port <follower> --id follower_so100 \
  --task "Pick up the hair serum and place it in the delivery zone" \
  --item "سيروم الشعر" --attempts 3
```

---

## 9. Repository structure

Organized into **`GUI/`** (frontend), **`backend/`** (server + logic), with the model
calls grouped under **`backend/agent/`** — one module per model ("agent 1, 2, …").

```
baseer/
├── GUI/
│   └── index.html              phone UI (tap-and-hold, audio in/out, VoiceOver-friendly)
├── backend/
│   ├── server.py               FastAPI host: /command-audio, /tts, /command (+ sessions, TTS cache)
│   ├── tools.py                action space: perceive_scene / deliver / say / ask (+ product registry)
│   ├── prompts.py              system prompt + hard rules
│   ├── normalize.py            ASR post-correction (diacritics + entity aliases)
│   ├── test_agent.py           offline agent tests (no API key needed)
│   ├── agent/                  ── the per-model "agents" ──
│   │   ├── fanar_base.py       shared Fanar transport (base URL, key, model ids, errors)
│   │   ├── agent1_reasoning.py  Fanar-C-2-27B  — the brain / decision-maker
│   │   ├── agent2_voice.py      Aura ASR + TTS — hear + speak Arabic
│   │   ├── agent3_vision.py     Fanar-Oryx     — identify + localize items
│   │   ├── agent4_grasp.py      SmolVLA        — localize → grasp → verify → retry → deliver
│   │   ├── agent5_yolo.py       YOLO-World     — alternative local eyes
│   │   └── orchestrator.py     the ReAct loop that drives agent1 to choose tools
│   └── robot/                  ── arm calibration + camera utilities ──
│       ├── calibrate_grasp.py        empty-vs-holding gripper thresholds → grasp_cfg.json
│       ├── calibrate_localization.py pixel → hover-pose map              → localization_map.json
│       ├── save_delivery_pose.py     fixed hand-off trajectory           → delivery_pose.json
│       ├── capture_policy_view.py    save exactly what the policy sees (debug the camera view)
│       ├── check_state_match.py      compare live joints to the training distribution
│       └── live_view.py / oryx_view.py / capture.py / match_camera.py
├── baseer_record/              record + train scripts, RECORDING_GUIDE.md
├── docs/index.html             project page (GitHub Pages)
└── README.md · requirements.txt · .env.example
```

---

## 10. Demo mode & future work

**What runs in the live demo (reliable):** voice → Aura ASR → Fanar agent → **Oryx announces** which serums are on the table → **policy grasps** the target (verified by torque + gripper width, with retry) → **scripted delivery** (auto lift-to-clear → set down at the zone) → Aura speaks the confirmation. The requested serum is placed in the grasp area.

**Object-precise localization is future work.** We built Oryx-guided pre-positioning (a calibrated pixel→arm-pose map + top-down approach) to pick the *right* item out of several, but with a **single RGB camera and no depth** the aim is a few cm off — not reliable enough for the demo. The principled fix is a **depth camera or hand-eye IK calibration** (a verified SO-100 URDF + `RobotKinematics`); the hooks are in the repo (`backend/robot/calibrate_localization.py`, `BASEER_PREREACH=1`).

### Future Fanar improvements
turn Fanar Oryx into a vision language action model. 


---

*Baseer — where the intelligence is genuinely Fanar's, the impact is dignified, and the system proves it by failing gracefully.*
