"""Voice Activity Detection using Silero VAD.

Used to decide whether to trigger analysis (hybrid mode) and to trim mic audio
before sending to Whisper.
"""
from __future__ import annotations

import numpy as np
import torch
from loguru import logger


class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        logger.info("Loading Silero VAD...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        (self.get_speech_timestamps, _, _, _, _) = self.utils
        logger.info("VAD loaded")

    def get_speech_duration(self, audio: np.ndarray) -> float:
        """Return total seconds of speech detected in the audio array."""
        if audio.size == 0:
            return 0.0
        tensor = torch.from_numpy(audio).float()
        try:
            timestamps = self.get_speech_timestamps(
                tensor, self.model, sampling_rate=self.sample_rate, threshold=self.threshold
            )
        except Exception as e:
            logger.warning("VAD failed: {}", e)
            return 0.0
        total_samples = sum(seg["end"] - seg["start"] for seg in timestamps)
        return total_samples / self.sample_rate

    def has_speech(self, audio: np.ndarray, min_duration: float = 1.0) -> bool:
        return self.get_speech_duration(audio) >= min_duration
