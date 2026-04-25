"""Test only the avatar overlay with a synthetic audio loop.
Run: python -m src.test_avatar
"""
import asyncio
import threading
import time
import numpy as np

from src.avatar import run_overlay_app, get_bridge


def fake_audio_loop():
    time.sleep(2)
    sample_rate = 24000
    while True:
        # Generate 3s of "talking" — modulated noise
        duration = 3.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        envelope = (np.sin(2 * np.pi * 4 * t) ** 2) * 0.4
        samples = (np.random.randn(len(t)).astype(np.float32) * envelope)
        get_bridge().speak.emit(samples, sample_rate)
        time.sleep(duration + 2)


if __name__ == "__main__":
    app, overlay = run_overlay_app()
    threading.Thread(target=fake_audio_loop, daemon=True).start()
    import sys
    sys.exit(app.exec())
