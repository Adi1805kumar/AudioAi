"""
audio_output.py — Thread 3: Speaker Playback Queue
────────────────────────────────────────────────────
Receives 24kHz 16-bit PCM audio chunks from the WebSocket manager,
buffers them, and plays them back smoothly. Supports barge-in:
stops immediately when the interrupt_event is set.

Design choices:
  • Dedicated daemon thread — never blocks the WebSocket event loop
  • queue.Queue for thread-safe chunk delivery
  • `playback_active` event lets Thread-1 know when AI is speaking
    (enables barge-in detection)
"""

import pyaudio
import threading
import queue
import numpy as np
from loguru import logger
from app.config import AUDIO_OUTPUT_SAMPLE_RATE, AUDIO_CHANNELS

_PLAYBACK_CHUNK = 2048   # frames per write — balance between latency & smoothness


class AudioOutput:
    """
    Thread 3 — consumes audio bytes from `playback_queue` and plays them
    through the default output device.

    Usage:
        output = AudioOutput(playback_queue, interrupt_event, playback_active)
        output.start()
    """

    def __init__(
        self,
        playback_queue: queue.Queue,
        interrupt_event: threading.Event,
        playback_active: threading.Event,
    ):
        self.playback_queue = playback_queue
        self.interrupt_event = interrupt_event   # set by Thread-1 on barge-in
        self.playback_active = playback_active   # we set this while speaking

        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._running = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="AudioOutput")

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=_PLAYBACK_CHUNK,
        )
        self._thread.start()
        logger.info("[Thread-3] Audio output started ({}Hz)", AUDIO_OUTPUT_SAMPLE_RATE)

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        logger.info("[Thread-3] Audio output stopped.")

    def enqueue(self, audio_bytes: bytes):
        """Called by WebSocket manager to queue incoming audio from Gemini."""
        self.playback_queue.put_nowait(audio_bytes)

    def clear_queue(self):
        """Drain the queue (used after barge-in interrupt)."""
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
            except queue.Empty:
                break

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run(self):
        while self._running:
            try:
                chunk = self.playback_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # ── Check for barge-in interrupt ───────────────────────────────
            if self.interrupt_event.is_set():
                logger.debug("[Thread-3] Interrupt received — clearing playback queue.")
                self.clear_queue()
                self.interrupt_event.clear()
                self.playback_active.clear()
                continue

            # ── Play the chunk ─────────────────────────────────────────────
            self.playback_active.set()
            try:
                self._stream.write(chunk)
            except OSError as e:
                logger.warning("[Thread-3] Stream write error: {}", e)

            # If queue is now empty, mark playback as done
            if self.playback_queue.empty():
                self.playback_active.clear()