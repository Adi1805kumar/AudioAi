"""
websocket_manager.py — Thread 2: Gemini Live API WebSocket Manager
───────────────────────────────────────────────────────────────────
Uses the NEW google-genai package (replaces deprecated google-generativeai).
"""

import asyncio
import threading
import queue
from loguru import logger

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_PROMPT
from app.tools import execute_tool

# ── Tool declarations for Gemini function-calling ────────────────────────────
TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="google_search",
        description=(
            "Search the web for current information: news, weather, "
            "documentation, prices, or any live data."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="The search query string.",
                )
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="search_documents",
        description=(
            "Search the user's local documents (PDFs, notes) "
            "for relevant information using RAG."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description="What to look for in the local documents.",
                )
            },
            required=["query"],
        ),
    ),
]


class WebSocketManager:
    """
    Thread 2 — manages the Gemini Live API session.

    Usage:
        manager = WebSocketManager(audio_queue, playback_queue, interrupt_event, playback_active)
        manager.start()
        manager.stop()
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        playback_queue: queue.Queue,
        interrupt_event: threading.Event,
        playback_active: threading.Event,
    ):
        self.audio_queue = audio_queue
        self.playback_queue = playback_queue
        self.interrupt_event = interrupt_event
        self.playback_active = playback_active

        self._loop: asyncio.AbstractEventLoop = None
        self._session = None
        self._running = False
        self._thread = threading.Thread(
            target=self._thread_entry, daemon=True, name="WebSocketManager"
        )

        self._client = genai.Client(api_key=GEMINI_API_KEY)

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread.start()
        logger.info("[Thread-2] WebSocket manager thread started.")

    def stop(self):
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        logger.info("[Thread-2] WebSocket manager stopped.")

    # ── Thread entry ───────────────────────────────────────────────────────────

    def _thread_entry(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        except Exception as e:
            logger.error("[Thread-2] Event loop crashed: {}", e)
        finally:
            self._loop.close()

    # ── Async core ─────────────────────────────────────────────────────────────

    async def _run(self):
        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Charon"
                    )
                )
            ),
        )

        backoff = 1
        while self._running:
            try:
                logger.info("[Thread-2] Connecting to Gemini Live API…")
                async with self._client.aio.live.connect(
                    model=GEMINI_MODEL, config=live_config
                ) as session:
                    self._session = session
                    backoff = 1
                    logger.success("[Thread-2] Connected to Gemini Live API ✓")
                    await asyncio.gather(
                        self._send_audio_loop(session),
                        self._receive_loop(session),
                    )
            except Exception as e:
                logger.warning(
                    "[Thread-2] Session error: {} — reconnecting in {}s", e, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _send_audio_loop(self, session):
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                chunk: bytes = await loop.run_in_executor(
                    None, lambda: self.audio_queue.get(timeout=0.05)
                )
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            if self.interrupt_event.is_set():
                continue

            await session.send(
                input=types.LiveClientRealtimeInput(
                    media_chunks=[
                        types.Blob(
                            data=chunk,
                            mime_type="audio/pcm;rate=16000",
                        )
                    ]
                )
            )

    async def _receive_loop(self, session):
        async for response in session.receive():

            # ── Audio bytes → playback queue ───────────────────────────────
            if response.data:
                self.playback_queue.put_nowait(response.data)
                continue

            # ── Server content (interruptions etc.) ───────────────────────
            if response.server_content:
                sc = response.server_content
                if sc.interrupted:
                    logger.debug("[Thread-2] Turn interrupted by Gemini.")
                    while not self.playback_queue.empty():
                        try:
                            self.playback_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.playback_active.clear()

            # ── Tool / function call ───────────────────────────────────────
            if response.tool_call:
                await self._handle_tool_call(session, response.tool_call)

    async def _handle_tool_call(self, session, tool_call):
        results = []
        for fn_call in tool_call.function_calls:
            logger.info("[Thread-2] Tool call: {} args={}", fn_call.name, fn_call.args)
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, execute_tool, fn_call.name, dict(fn_call.args)
                )
            except Exception as e:
                result = {"error": str(e)}

            results.append(
                types.FunctionResponse(
                    id=fn_call.id,
                    name=fn_call.name,
                    response={"result": result},
                )
            )

        await session.send(
            input=types.LiveClientToolResponse(function_responses=results)
        )