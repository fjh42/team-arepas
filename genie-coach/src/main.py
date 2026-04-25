"""Genie Coach entry point.

Runs the Qt overlay on the main thread and the async orchestrator on a worker
thread (Qt and asyncio don't share an event loop cleanly otherwise).
"""
from __future__ import annotations

import asyncio
import sys
import threading

from loguru import logger

from .avatar import run_overlay_app
from .orchestrator import Orchestrator


def start_orchestrator_thread():
    """Run the async orchestrator inside a dedicated thread."""
    def thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        orch = Orchestrator()
        try:
            loop.run_until_complete(orch.run())
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.exception("Orchestrator crashed: {}", e)
        finally:
            loop.close()

    t = threading.Thread(target=thread_target, daemon=True, name="orchestrator")
    t.start()
    return t


def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

    logger.info("Starting Genie Coach")

    # 1. Boot Qt overlay (main thread)
    app, overlay = run_overlay_app()

    # 2. Start orchestrator in worker thread
    start_orchestrator_thread()

    # 3. Run Qt event loop (blocks until window closed)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
