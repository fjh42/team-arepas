"""Speech-to-text using faster-whisper.

Runs locally on GPU (or CPU as fallback). Used to transcribe the player's mic
audio so the LLM can react to what they said.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from faster_whisper import WhisperModel
from loguru import logger

from .config import get_config


class Transcriber:
    def __init__(self):
        cfg = get_config()
        stt_cfg = cfg.stt
        logger.info("Loading Whisper model '{}' on {}...", stt_cfg["model"], stt_cfg["device"])
        try:
            self.model = WhisperModel(
                stt_cfg["model"],
                device=stt_cfg["device"],
                compute_type=stt_cfg["compute_type"],
            )
        except Exception as e:
            logger.warning("GPU load failed ({}), falling back to CPU", e)
            self.model = WhisperModel(stt_cfg["model"], device="cpu", compute_type="int8")
        self.language = stt_cfg["language"]
        logger.info("Whisper ready")

    async def transcribe(self, audio_path: Path) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio_path)

    def _transcribe_sync(self, audio_path: Path) -> str:
        try:
            segments, _info = self.model.transcribe(
                str(audio_path),
                language=self.language,
                beam_size=1,           # greedy for speed
                vad_filter=True,        # built-in VAD trims silence
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text
        except Exception as e:
            logger.error("Transcription failed: {}", e)
            return ""
