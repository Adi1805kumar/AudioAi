"""
audio_input.py — Thread 1: Continuous Microphone Capture
─────────────────────────────────────────────────────────
Captures system/mic audio at 16kHz 16-bit PCM mono and pushes
raw chunks into a thread-safe queue consumed by the WebSocket manager.

Key design decisions:
  • Uses PyAudio callback mode → near-zero latency, no dropped frames
  • Barge-in detection: if the playback thread is active, it signals
    an interrupt via the shared `interrupt_event`
  • VAD (Voice Activity Detection) is applied to avoid sending silence
    to Gemini (saves tokens & latency)
"""

import pyaudio
import threading
import queue
import numpy as np
from loguru import logger
from app.config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_CHUNK_SIZE,
)

# ── Simple energy-based VAD ───────────────────────────────────────────────────
_SILENCE_THRESHOLD = 300   # RMS amplitude below this → silence
_SILENCE_FRAMES_MAX = 30   # ~0.3 s of silence before stopping stream to Gemini


def _is_speech(data: bytes) -> bool:
    """Return True if the audio chunk contains speech (not silence)."""
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples ** 2))
    return rms > _SILENCE_THRESHOLD


# ── AudioIngestion ─────────────────────────────────────────────────────────────
class AudioIngestion:
    """
    Thread 1 — opens the default microphone and streams audio chunks
    into `audio_queue` for the WebSocket manager to consume.

    Usage:
        ingestion = AudioIngestion(audio_queue, interrupt_event)
        ingestion.start()
        # ... later ...
        ingestion.stop()
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        interrupt_event: threading.Event,
        playback_active: threading.Event,
    ):
        self.audio_queue = audio_queue
        self.interrupt_event = interrupt_event   # signals barge-in to playback
        self.playback_active = playback_active   # set by AudioOutput when speaking

        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._running = False
        self._silence_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="AudioIngestion")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread.start()
        logger.info("[Thread-1] Audio ingestion started ({}Hz, {}ch, chunk={})",
                    AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_CHUNK_SIZE)

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        logger.info("[Thread-1] Audio ingestion stopped.")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(self):
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK_SIZE,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

        # Keep thread alive while stream is open
        while self._running and self._stream.is_active():
            threading.Event().wait(0.1)

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback — called in a separate C thread, must be fast."""
        if not self._running:
            return (None, pyaudio.paComplete)

        # ── Barge-in detection ─────────────────────────────────────────────
        if self.playback_active.is_set() and _is_speech(in_data):
            logger.debug("[Thread-1] Barge-in detected — signalling interrupt.")
            self.interrupt_event.set()   # Thread-3 will stop playback

        # ── VAD: only forward speech frames ───────────────────────────────
        if _is_speech(in_data):
            self._silence_count = 0
            self.audio_queue.put_nowait(in_data)
        else:
            self._silence_count += 1
            if self._silence_count <= _SILENCE_FRAMES_MAX:
                # Send a brief tail of silence so Gemini knows speech ended
                self.audio_queue.put_nowait(in_data)

        return (None, pyaudio.paContinue)