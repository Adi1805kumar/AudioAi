"""
main.py — FastAPI Application Entry Point
──────────────────────────────────────────
Bootstraps the three-thread architecture:
  Thread 1: AudioIngestion    — mic → audio_queue
  Thread 2: WebSocketManager  — audio_queue → Gemini → playback_queue
  Thread 3: AudioOutput       — playback_queue → speaker

Also exposes REST + WebSocket endpoints for:
  • Health check
  • PDF document ingestion (drag-and-drop or API call)
  • Status dashboard
  • Manual text query (for testing without mic)
"""

import queue
import threading
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger

from app.config import HOST, PORT
from app.audio_input import AudioIngestion
from app.audio_output import AudioOutput
from app.websocket_manager import WebSocketManager
from app.rag_engine import ingest_pdf, ingest_folder, search as rag_search

# ── Shared state ──────────────────────────────────────────────────────────────
audio_queue: queue.Queue = queue.Queue(maxsize=100)     # Thread 1 → Thread 2
playback_queue: queue.Queue = queue.Queue(maxsize=200)  # Thread 2 → Thread 3
interrupt_event: threading.Event = threading.Event()    # Thread 1 signals Thread 3
playback_active: threading.Event = threading.Event()    # Thread 3 signals Thread 1

# ── Service instances ──────────────────────────────────────────────────────────
audio_ingestion: AudioIngestion = None
audio_output: AudioOutput = None
ws_manager: WebSocketManager = None

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hands-Free AI Desktop Assistant",
    description="Local Python service: PyAudio + Gemini Live API + RAG",
    version="1.0.0",
)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global audio_ingestion, audio_output, ws_manager

    logger.info("═" * 60)
    logger.info("  Hands-Free AI Desktop Assistant — Starting up")
    logger.info("═" * 60)

    # Auto-ingest any PDFs in the docs/ folder
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ingest_folder, "./docs")

    # Thread 3: start first so playback is ready
    audio_output = AudioOutput(playback_queue, interrupt_event, playback_active)
    audio_output.start()

    # Thread 2: WebSocket manager (connects to Gemini)
    ws_manager = WebSocketManager(audio_queue, playback_queue, interrupt_event, playback_active)
    ws_manager.start()

    # Thread 1: mic capture (start last so Gemini is ready to receive)
    audio_ingestion = AudioIngestion(audio_queue, interrupt_event, playback_active)
    audio_ingestion.start()

    logger.success("All threads running. Assistant is listening… 🎤")


@app.on_event("shutdown")
async def shutdown():
    if audio_ingestion:
        audio_ingestion.stop()
    if ws_manager:
        ws_manager.stop()
    if audio_output:
        audio_output.stop()
    logger.info("Assistant shut down cleanly.")


# ── REST Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Quick health check — useful for testing without audio."""
    return {
        "status": "ok",
        "threads": {
            "audio_ingestion": audio_ingestion._thread.is_alive() if audio_ingestion else False,
            "ws_manager": ws_manager._thread.is_alive() if ws_manager else False,
            "audio_output": audio_output._thread.is_alive() if audio_output else False,
        },
        "queues": {
            "audio_queue_size": audio_queue.qsize(),
            "playback_queue_size": playback_queue.qsize(),
        },
        "events": {
            "interrupt": interrupt_event.is_set(),
            "playback_active": playback_active.is_set(),
        },
    }


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload a PDF to be indexed for RAG retrieval.
    The assistant will be able to answer questions about it immediately.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = Path("./docs") / file.filename
    save_path.parent.mkdir(exist_ok=True)

    contents = await file.read()
    save_path.write_bytes(contents)

    loop = asyncio.get_event_loop()
    chunks_added = await loop.run_in_executor(None, ingest_pdf, str(save_path))

    return JSONResponse({
        "status": "ingested",
        "file": file.filename,
        "chunks_added": chunks_added,
    })


@app.get("/search")
async def search_docs(q: str):
    """
    Debug endpoint: test RAG search without voice.
    Example: GET /search?q=what is the refund policy
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, rag_search, q)
    return {"query": q, "result": result}


@app.get("/status")
async def status():
    """Human-readable status for the terminal dashboard."""
    return {
        "listening": audio_ingestion._thread.is_alive() if audio_ingestion else False,
        "connected_to_gemini": ws_manager._thread.is_alive() if ws_manager else False,
        "speaking": playback_active.is_set(),
        "interrupted": interrupt_event.is_set(),
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=False,   # Never use reload=True in production (breaks threads)
        log_level="info",
    )