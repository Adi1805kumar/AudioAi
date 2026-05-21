# 🎤 Hands-Free AI Desktop Assistant

A local Python background service that gives you a **voice-first AI assistant** while you stay focused on your screen — no clicking, no typing.

Built with **Gemini Live API** (real-time multimodal), **PyAudio** (low-latency audio), and **ChromaDB** (local RAG for your PDFs).

---

## Architecture

```
USER WINDOWS (PDF Reader / IDE / Chrome)
          │  Continuous background capture
          ▼
┌─────────────────────────────────────────────────────┐
│           LOCAL PYTHON SERVICE (FastAPI)            │
│                                                     │
│  Thread 1: AudioIngestion                           │
│  ┌──────────────────────┐   Audio Bytes (16kHz PCM) │
│  │ PyAudio mic stream   │ ─────────────────────────►│
│  │ VAD + silence filter │                           │
│  └────────┬─────────────┘                           │
│           │ Barge-in interrupt                      │
│           ▼                                         │
│  Thread 3: AudioOutput  ◄── Audio Bytes (24kHz PCM) │
│  ┌──────────────────────┐                           │
│  │ Buffer + queue       │                           │
│  │ Barge-in stop        │                           │
│  └──────────────────────┘                           │
│                                                     │
│  Thread 2: WebSocketManager ◄──► Gemini Live API    │
│  ┌──────────────────────┐    Bidirectional WebSocket│
│  │ asyncio EventLoop    │                           │
│  │ Send/Receive audio   │◄── Tools: Google Search   │
│  │ Function calling     │◄── RAG: ChromaDB + PDFs   │
│  └──────────────────────┘                           │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- `portaudio` (required by PyAudio)

```bash
# macOS
brew install portaudio

# Ubuntu / Debian
sudo apt-get install portaudio19-dev python3-dev

# Windows
# PyAudio ships with PortAudio — no extra step needed
```

### 2. Clone & Install

```bash
git clone <your-repo>
cd desktop_ai_assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
# (Get it free at https://aistudio.google.com)
```

### 4. Add Documents (Optional)

Drop any PDF files into the `docs/` folder. They will be auto-indexed on startup and searchable via voice ("Hey, what does my contract say about refunds?").

### 5. Run

**Terminal 1 — Start the assistant:**
```bash
python -m app.main
# or:
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Live dashboard (optional):**
```bash
python dashboard.py
```

---

## Development Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Microphone capture (Thread 1) | ✅ |
| 2 | Gemini Live API connection (Thread 2) | ✅ |
| 3 | Audio playback (Thread 3) | ✅ |
| 4 | Barge-in / interrupt | ✅ |
| 5 | PDF RAG (ChromaDB) | ✅ |
| 6 | Google Search tool calling | ✅ |
| 7 | Latency optimization | 🔄 Ongoing |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Full system health (threads, queues, events) |
| GET | `/status` | Simple listening/speaking status |
| POST | `/ingest` | Upload a PDF to index for RAG |
| GET | `/search?q=...` | Debug RAG search without voice |

### Upload a PDF via curl:
```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F "file=@/path/to/your/document.pdf"
```

---

## Configuration

All settings live in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | **Required.** From Google AI Studio |
| `SERPAPI_KEY` | — | Optional. For live web search |
| `GEMINI_MODEL` | `gemini-2.0-flash-live-001` | Gemini Live model |
| `AUDIO_SAMPLE_RATE` | `16000` | Mic input sample rate (Hz) |
| `AUDIO_OUTPUT_SAMPLE_RATE` | `24000` | Speaker output rate (Hz) |
| `AUDIO_CHUNK_SIZE` | `1024` | Frames per mic read |
| `CHROMA_DB_PATH` | `./vector_db` | Where ChromaDB stores data |
| `RAG_TOP_K` | `5` | Number of doc chunks to retrieve |

---

## Biggest Challenges & Solutions

### Echo Cancellation
**Problem:** AI voice re-enters the microphone.  
**Solution:** Use headphones. The VAD threshold in `audio_input.py` (`_SILENCE_THRESHOLD`) can be tuned if needed.

### Audio Latency
**Problem:** Delay between speaking and AI response.  
**Solution:** PyAudio callback mode (not blocking reads) + chunked streaming to Gemini.

### Thread Safety
**Problem:** Three threads sharing data.  
**Solution:** All cross-thread communication uses `queue.Queue` (thread-safe) and `threading.Event`.

### WebSocket Stability
**Problem:** Network drops disconnect Gemini session.  
**Solution:** Exponential backoff reconnection loop in `websocket_manager.py`.

---

## Project Structure

```
desktop_ai_assistant/
├── app/
│   ├── main.py               # FastAPI app + startup orchestration
│   ├── config.py             # All settings from .env
│   ├── audio_input.py        # Thread 1: mic capture + VAD
│   ├── audio_output.py       # Thread 3: speaker playback + barge-in
│   ├── websocket_manager.py  # Thread 2: Gemini Live API session
│   ├── rag_engine.py         # PDF ingestion + vector search
│   └── tools.py              # Google Search + RAG tool handlers
├── dashboard.py              # Rich terminal live dashboard
├── docs/                     # Drop PDFs here for RAG
├── vector_db/                # ChromaDB persistent storage (auto-created)
├── .env.example              # Copy to .env and fill in keys
├── requirements.txt
└── README.md
```