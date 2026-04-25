"""Transparent always-on-top avatar overlay with audio-driven lip sync.

Uses PyQt6 to render a frameless transparent window in a screen corner. The
genie image is shown with a "mouth" overlay that scales vertically based on
the current audio amplitude — a poor-man's lip sync that runs at 60fps with
near-zero CPU cost. Replace _render_with_mouth() with a Wav2Lip pipeline for
realism.
"""
from __future__ import annotations

import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np
import pyaudio
from loguru import logger
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPainter, QPixmap, QColor, QGuiApplication
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from .config import get_config

# Bridge non-Qt threads → Qt main thread
class _AudioBridge(QObject):
    speak = pyqtSignal(np.ndarray, int)


_bridge: _AudioBridge | None = None


def get_bridge() -> _AudioBridge:
    global _bridge
    if _bridge is None:
        _bridge = _AudioBridge()
    return _bridge


class GenieOverlay(QWidget):
    def __init__(self):
        super().__init__()
        cfg = get_config()
        self.cfg = cfg
        self.size_px = cfg.avatar["size"]

        # Frameless, transparent, always-on-top, click-through
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setFixedSize(self.size_px, self.size_px)

        # Position in chosen corner
        self._position_window(cfg.avatar["position"])

        # Load base image
        img_path = Path(__file__).resolve().parent.parent / cfg.avatar["image_path"]
        if img_path.exists():
            self.base_pixmap = QPixmap(str(img_path)).scaled(
                self.size_px, self.size_px,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            logger.warning("Avatar image not found at {}, using placeholder", img_path)
            self.base_pixmap = self._make_placeholder()

        # Audio playback state
        self.current_audio: np.ndarray = np.zeros(0, dtype=np.float32)
        self.audio_sample_rate: int = 24000
        self.audio_position: int = 0
        self.current_amplitude: float = 0.0
        self._audio_lock = threading.Lock()

        # Smoothed amplitude for nicer mouth motion
        self._amp_history = deque(maxlen=4)

        # Render timer (60 fps)
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.update)
        self.render_timer.start(16)

        # PyAudio output stream (lazily created)
        self._pa = pyaudio.PyAudio()
        self._stream: pyaudio.Stream | None = None

        # Connect bridge signal
        get_bridge().speak.connect(self.play_audio)

    def _position_window(self, corner: str):
        screen = QGuiApplication.primaryScreen().geometry()
        margin = 20
        if corner == "top-right":
            x = screen.width() - self.size_px - margin
            y = margin
        elif corner == "top-left":
            x = margin
            y = margin
        elif corner == "bottom-right":
            x = screen.width() - self.size_px - margin
            y = screen.height() - self.size_px - margin
        else:  # bottom-left
            x = margin
            y = screen.height() - self.size_px - margin
        self.move(x, y)

    def _make_placeholder(self) -> QPixmap:
        pix = QPixmap(self.size_px, self.size_px)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(80, 30, 200, 220))
        p.drawEllipse(10, 10, self.size_px - 20, self.size_px - 20)
        p.setPen(QColor(255, 255, 255))
        p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "GENIE")
        p.end()
        return pix

    # ---- audio playback (called via Qt signal so we're on main thread) ----
    def play_audio(self, samples: np.ndarray, sample_rate: int):
        """Start playing the given audio buffer; amplitude drives mouth."""
        if samples.size == 0:
            return
        with self._audio_lock:
            self.current_audio = samples.astype(np.float32)
            self.audio_sample_rate = sample_rate
            self.audio_position = 0

        # Start an output stream if needed
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=sample_rate,
            output=True,
            frames_per_buffer=1024,
            stream_callback=self._audio_callback,
        )
        self._stream.start_stream()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        with self._audio_lock:
            start = self.audio_position
            end = min(start + frame_count, self.current_audio.size)
            chunk = self.current_audio[start:end]
            self.audio_position = end

        if chunk.size < frame_count:
            chunk = np.concatenate([chunk, np.zeros(frame_count - chunk.size, dtype=np.float32)])
            flag = pyaudio.paComplete
        else:
            flag = pyaudio.paContinue

        # Update amplitude (RMS of chunk)
        rms = float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0
        self._amp_history.append(rms)
        self.current_amplitude = sum(self._amp_history) / len(self._amp_history)

        return (chunk.tobytes(), flag)

    # ---- rendering ----
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw base avatar
        painter.drawPixmap(0, 0, self.base_pixmap)

        # Draw an animated "mouth" rectangle that scales with amplitude
        # Tweak coordinates for your specific genie image
        amp = min(1.0, self.current_amplitude * 8.0)  # boost for visibility
        if amp > 0.01:
            mouth_w = int(self.size_px * 0.18)
            mouth_h_max = int(self.size_px * 0.10)
            mouth_h = max(2, int(mouth_h_max * amp))
            cx = self.size_px // 2
            cy = int(self.size_px * 0.72)  # adjust for your image
            painter.setBrush(QColor(20, 0, 0, 230))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                cx - mouth_w // 2, cy - mouth_h // 2,
                mouth_w, mouth_h, 6, 6,
            )

    def closeEvent(self, event):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
        super().closeEvent(event)


def run_overlay_app() -> tuple[QApplication, GenieOverlay]:
    app = QApplication.instance() or QApplication(sys.argv)
    overlay = GenieOverlay()
    overlay.show()
    return app, overlay
