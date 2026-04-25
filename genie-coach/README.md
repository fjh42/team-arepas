# Genie Coach 🧞 — Real-Time Gaming AI Companion

A trash-talking AI genie that watches you play, listens to your mic, and roasts/coaches you via a lip-synced avatar overlay.

## Architecture

```
Screen + Mic → 30s rolling buffer → Gemini Flash (vision)
                                  ↓
                            Whisper (your speech)
                                  ↓
                       LLM persona response (snarky genie)
                                  ↓
                         ElevenLabs TTS (streaming)
                                  ↓
                    Lip-synced avatar (transparent overlay)
```

## Hardware Requirements

- **Minimum:** RTX 3060 (8GB VRAM), 16GB RAM, decent CPU
- **Recommended:** RTX 4070+ (12GB+ VRAM), 32GB RAM
- Windows 10/11, Linux, or macOS (avatar overlay works best on Windows/Linux)

## Setup

### 1. Install system dependencies

**Windows:**
```powershell
# Install FFmpeg: https://www.gyan.dev/ffmpeg/builds/ → add to PATH
# Install OBS Studio: https://obsproject.com (enable WebSocket in Tools → WebSocket Settings)
```

**Linux:**
```bash
sudo apt install ffmpeg portaudio19-dev python3-pyqt6
```

**macOS:**
```bash
brew install ffmpeg portaudio
```

### 2. Python environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API keys

Copy `.env.example` to `.env` and fill in:
- `GEMINI_API_KEY` — get from https://aistudio.google.com/apikey
- `ELEVENLABS_API_KEY` — get from https://elevenlabs.io
- `ELEVENLABS_VOICE_ID` — pick a voice from your ElevenLabs library

### 4. Add a genie avatar image

Drop a PNG of your genie character into `assets/genie.png` (square, transparent background works best, ~512x512). The lip-sync uses this as the base.

### 5. Run it

```bash
python -m src.main
```

A transparent overlay window appears in the top-right of your screen. Start your game and the genie will start talking after the first 30s buffer fills.

## Configuration

Edit `config/settings.yaml` to tune:
- Capture interval (how often clips are analyzed)
- Trigger mode (time-based vs. voice-activated)
- Persona prompt (how mean is your genie?)
- Avatar position and size

## Tips

- Start with cloud APIs (Gemini + ElevenLabs) before going fully local — much easier to debug
- The first run downloads Whisper + VAD models (~500MB)
- If you hear yourself in the genie's audio, lower mic gain or use a directional mic
- For maximum performance, run the game on one GPU and inference on another (if you have two)

## Troubleshooting

- **No audio from mic in clips:** check `config/settings.yaml` → `audio.input_device_index`
- **Avatar doesn't appear:** Linux Wayland users need to switch to X11 for transparent overlays
- **Gemini timeouts:** reduce `capture.video_width` and `capture.fps` in settings
