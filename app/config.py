"""
config.py — Central configuration loaded from .env
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Copy .env.example → .env and add your key from https://aistudio.google.com"
    )

# ── Audio ─────────────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_OUTPUT_SAMPLE_RATE: int = int(os.getenv("AUDIO_OUTPUT_SAMPLE_RATE", "24000"))
AUDIO_CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))
AUDIO_CHUNK_SIZE: int = int(os.getenv("AUDIO_CHUNK_SIZE", "1024"))

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash-live-001")

SYSTEM_PROMPT: str = """
You are a hands-free AI desktop assistant. The user is focused on their screen
(reading PDFs, coding, or browsing). Speak concisely and naturally — you are a
voice-first assistant. When the user asks about documents or files, relevant
context will be injected automatically. Use tools for live information like
weather, news, or code documentation. Keep responses short unless asked to
elaborate.
"""

# ── RAG ───────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./vector_db")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

# ── Server ────────────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "8000"))