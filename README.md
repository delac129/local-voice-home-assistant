# local-voice-home-assistant
A fully local, voice-controlled home automation assistant. A fine-tuned **GPT-2** model maps
natural-language commands — like *"good night"* or *"it's too cold"* — to device actions
(lights, locks, temperature, blinds). The model is served through a **Flask** gateway, with a
**React + Web Speech API** frontend for voice and text input.

Everything runs **on-device** with no cloud dependency, keeping commands and data inside the home.

---

## What it does

You speak or type a natural-language command, and the assistant maps it to one or more device
actions. It handles direct commands, casual phrasing, and multi-action routines:

| You say | It does |
|---|---|
| "turn off the lights" | `lights_off` |
| "yo kill the lights" | `lights_off` |
| "it's too cold" | `increase_temp` |
| "good night" | `lights_off, lock_door, decrease_temp` |
| "movie time" | `dim_lights, close_blinds` |
| "I'm leaving" | `lights_off, lock_door` |

The interface shows each command, the resulting action(s), the input source (voice or text),
and the response latency.

---

## How it works

**1. Intent model (`app.py`)**
GPT-2 is fine-tuned on a small set of command → action examples so it learns to translate
free-form language into a fixed vocabulary of device actions. The fine-tuning runs locally on CPU.

**2. Gateway (Flask)**
A lightweight Flask API exposes two endpoints:
- `POST /command` — takes a natural-language command, returns the predicted action(s) and latency
- `GET /health` — reports gateway status

**3. Frontend (`frontend.html`)**
A React single-page UI (via CDN, no build step) sends commands to the gateway and renders the
results. Voice input uses the browser's **Web Speech API**, so speech recognition runs in the
browser with no third-party service.

```
Voice / Text  ->  React UI  ->  Flask gateway  ->  fine-tuned GPT-2  ->  action(s)
```

---

## Tech stack

- **Python** — Flask, Flask-CORS
- **ML** — Hugging Face Transformers (GPT-2), PyTorch
- **Frontend** — React, Web Speech API

---

## Running it locally

**Backend**
```bash
pip install -r requirements.txt
python app.py
```
On first run it downloads GPT-2 and fine-tunes it locally (CPU is fine), then starts the gateway
at `http://localhost:5000`.

**Frontend**
Open `frontend.html` in Chrome (voice input requires a Chromium-based browser). The status badge
turns green once it connects to the gateway, then type or tap the mic to send a command.

---

## Notes

This is a proof-of-concept built to explore local-first, natural-language home control — turning
spoken intent into device actions without sending anything to the cloud. The action vocabulary
and training examples are intentionally small and easy to extend in `app.py`.
