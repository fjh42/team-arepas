"""ElevenLabs streaming TTS.

Returns raw PCM audio chunks the avatar/audio-output can consume.
"""
from __future__ import annotations

import asyncio
import io
from typing import AsyncIterator

import numpy as np
from loguru import logger
from pydub import AudioSegment

from .config import get_config


class TTSEngine:
    def __init__(self):
        cfg = get_config()
        self.cfg = cfg
        if cfg.tts["provider"] == "elevenlabs":
            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs(api_key=cfg.elevenlabs_api_key)
            self.voice_id = cfg.elevenlabs_voice_id
            self.model_id = cfg.tts["model"]
        else:
            raise NotImplementedError(f"TTS provider {cfg.tts['provider']} not yet wired")

    async def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Generate audio from text. Returns (samples_float32_mono, sample_rate)."""
        if not text.strip():
            return np.zeros(0, dtype=np.float32), 24000

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        try:
            audio_iter = self.client.text_to_speech.convert(
                voice_id=self.voice_id,
                model_id=self.model_id,
                text=text,
                voice_settings={
                    "stability": self.cfg.tts["stability"],
                    "similarity_boost": self.cfg.tts["similarity_boost"],
                },
                output_format="mp3_44100_128",
            )
            mp3_bytes = b"".join(audio_iter)
        except Exception as e:
            logger.error("TTS failed: {}", e)
            return np.zeros(0, dtype=np.float32), 44100

        # Decode mp3 → numpy
        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3").set_channels(1)
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        # Normalize int16 → float32 [-1, 1]
        if seg.sample_width == 2:
            samples /= 32768.0
        return samples, seg.frame_rate
