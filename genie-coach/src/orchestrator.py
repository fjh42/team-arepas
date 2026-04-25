"""Orchestrator: runs the capture → analyze → speak loop.

Runs in its own thread so the Qt overlay can own the main thread.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from loguru import logger

from .avatar import get_bridge
from .capture import CaptureBuffer
from .config import get_config
from .llm import GenieLLM
from .tts import TTSEngine
from .transcribe import Transcriber
from .vad import VoiceActivityDetector


class Orchestrator:
    def __init__(self):
        self.cfg = get_config()
        self.capture = CaptureBuffer()
        self.vad = VoiceActivityDetector(sample_rate=self.cfg.audio["sample_rate"])
        self.transcriber = Transcriber()
        self.llm = GenieLLM()
        self.tts = TTSEngine()
        self.work_dir = Path(tempfile.mkdtemp(prefix="genie_"))
        self._busy = False  # prevent overlapping cycles
        logger.info("Orchestrator initialized; work dir: {}", self.work_dir)

    async def run(self):
        await self.capture.start()
        # Wait for buffer to fill
        warmup = self.cfg.capture["clip_duration"]
        logger.info("Warming up buffer for {}s...", warmup)
        await asyncio.sleep(warmup)

        interval = self.cfg.capture["interval"]
        try:
            while True:
                if not self._busy:
                    asyncio.create_task(self._run_cycle())
                await asyncio.sleep(interval)
        finally:
            await self.capture.stop()
            shutil.rmtree(self.work_dir, ignore_errors=True)

    async def _run_cycle(self):
        self._busy = True
        try:
            # 1. Trigger check
            if not self._should_trigger():
                logger.debug("Skipping cycle (no trigger)")
                return

            # 2. Snapshot buffers
            video_path, audio_path = self.capture.snapshot(self.work_dir)

            # 3. Run STT and LLM in parallel where possible
            transcript_task = asyncio.create_task(self.transcriber.transcribe(audio_path))
            transcript = await transcript_task
            if transcript:
                logger.info("Player said: {}", transcript)

            # 4. LLM reaction (this dominates latency)
            response = await self.llm.react(video_path, transcript)
            if not response:
                logger.debug("Genie chose silence")
                return
            logger.success("Genie: {}", response)

            # 5. TTS synthesis
            samples, sample_rate = await self.tts.synthesize(response)
            if samples.size == 0:
                return

            # 6. Send to overlay (Qt signal hops to main thread)
            get_bridge().speak.emit(samples, sample_rate)

            # 7. Cleanup
            try:
                video_path.unlink(missing_ok=True)
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as e:
            logger.exception("Cycle failed: {}", e)
        finally:
            self._busy = False

    def _should_trigger(self) -> bool:
        mode = self.cfg.trigger["mode"]
        if mode == "time":
            return True
        audio = self.capture.get_audio_array()
        speech_dur = self.vad.get_speech_duration(audio)
        if mode == "voice":
            return speech_dur >= self.cfg.trigger["voice_min_duration_sec"]
        # "hybrid": always trigger, but log voice activity
        logger.debug("Speech detected in buffer: {:.1f}s", speech_dur)
        return True
