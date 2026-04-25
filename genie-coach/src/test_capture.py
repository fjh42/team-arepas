"""Test only screen+audio capture. Saves a 30s clip to ./test_output/.
Run: python -m src.test_capture
"""
import asyncio
from pathlib import Path

from src.capture import CaptureBuffer


async def main():
    buf = CaptureBuffer()
    await buf.start()
    print("Capturing 30s...")
    await asyncio.sleep(30)
    out = Path("./test_output")
    video, audio = buf.snapshot(out)
    await buf.stop()
    print(f"Saved {video} and {audio}")


if __name__ == "__main__":
    asyncio.run(main())
