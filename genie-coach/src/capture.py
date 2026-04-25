"""Screen + microphone capture with rolling buffers.

Continuously captures the screen at low fps and the mic, keeping the last N
seconds in memory. snapshot() returns the current buffer contents as files
suitable for Gemini upload.
"""
from __future__ import annotations

import asyncio
import io
import time
import wave
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np
import sounddevice as sd
from loguru import logger

from .config import get_config


class CaptureBuffer:
    def __init__(self):
        cfg = get_config()
        self.cfg = cfg
        self.duration = cfg.capture["clip_duration"]
        self.fps = cfg.capture["fps"]
        self.width = cfg.capture["video_width"]
        self.height = cfg.capture["video_height"]
        self.monitor_idx = cfg.capture["monitor"]

        self.sample_rate = cfg.audio["sample_rate"]
        self.channels = cfg.audio["channels"]
        self.input_device = cfg.audio["input_device_index"]

        # Rolling buffers
        max_frames = self.duration * self.fps
        self.video_buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=max_frames)
        max_audio_samples = self.duration * self.sample_rate
        self.audio_buffer: deque[float] = deque(maxlen=max_audio_samples)

        self._running = False
        self._video_task: Optional[asyncio.Task] = None
        self._audio_stream: Optional[sd.InputStream] = None

    async def start(self):
        self._running = True
        self._video_task = asyncio.create_task(self._capture_video_loop())
        self._start_audio_stream()
        logger.info("Capture started (video {}fps, audio {}Hz)", self.fps, self.sample_rate)

    async def stop(self):
        self._running = False
        if self._video_task:
            self._video_task.cancel()
        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
        logger.info("Capture stopped")

    def _start_audio_stream(self):
        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("Audio status: {}", status)
            # Flatten to mono if needed and extend ring buffer
            mono = indata[:, 0] if indata.ndim > 1 else indata
            self.audio_buffer.extend(mono.tolist())

        self._audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.input_device,
            callback=callback,
            blocksize=int(self.sample_rate * 0.05),  # 50ms blocks
        )
        self._audio_stream.start()

    async def _capture_video_loop(self):
        """Capture frames at target fps using mss (very fast)."""
        loop = asyncio.get_event_loop()
        sct = mss.mss()
        monitor = sct.monitors[self.monitor_idx]
        frame_interval = 1.0 / self.fps

        while self._running:
            t0 = time.time()
            # Run blocking grab in executor
            frame = await loop.run_in_executor(None, self._grab_frame, sct, monitor)
            if frame is not None:
                self.video_buffer.append((t0, frame))
            elapsed = time.time() - t0
            await asyncio.sleep(max(0, frame_interval - elapsed))

    def _grab_frame(self, sct, monitor) -> np.ndarray | None:
        try:
            raw = sct.grab(monitor)
            img = np.array(raw)  # BGRA
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img = cv2.resize(img, (self.width, self.height))
            return img
        except Exception as e:
            logger.error("Frame grab failed: {}", e)
            return None

    def snapshot(self, output_dir: Path) -> tuple[Path, Path]:
        """Write current buffer to mp4 + wav. Returns (video_path, audio_path)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        video_path = output_dir / f"clip_{ts}.mp4"
        audio_path = output_dir / f"clip_{ts}.wav"

        # Write video
        frames = list(self.video_buffer)
        if not frames:
            raise RuntimeError("Video buffer empty — capture not running long enough")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (self.width, self.height))
        for _, frame in frames:
            writer.write(frame)
        writer.release()

        # Write audio
        samples = np.array(list(self.audio_buffer), dtype=np.float32)
        # Convert float32 [-1, 1] to int16 PCM
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(audio_path), "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())

        logger.debug("Snapshot: {} ({} frames), {} ({:.1f}s audio)",
                     video_path.name, len(frames), audio_path.name, len(samples) / self.sample_rate)
        return video_path, audio_path

    def get_audio_array(self) -> np.ndarray:
        """Return current mic audio buffer as numpy array (for VAD/Whisper)."""
        return np.array(list(self.audio_buffer), dtype=np.float32)
