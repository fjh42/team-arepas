"""Gemini Flash LLM client.

Uploads the gameplay clip + mic transcript and gets back the genie's reaction.
Uses the Files API for efficient video upload.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import google.generativeai as genai
from loguru import logger

from .config import get_config


class GenieLLM:
    def __init__(self):
        cfg = get_config()
        genai.configure(api_key=cfg.gemini_api_key)
        self.cfg = cfg
        self.model = genai.GenerativeModel(
            model_name=cfg.llm["model"],
            system_instruction=cfg.persona,
        )
        # Keep a short rolling history so the genie has continuity
        self.history: list[dict] = []
        self.max_history = 6  # last 3 exchanges

    async def react(self, video_path: Path, mic_transcript: str) -> str:
        """Send clip + transcript, return genie's response (or empty if SILENT)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._react_sync, video_path, mic_transcript)

    def _react_sync(self, video_path: Path, mic_transcript: str) -> str:
        # Upload video via Files API
        try:
            uploaded = genai.upload_file(path=str(video_path), mime_type="video/mp4")
        except Exception as e:
            logger.error("Video upload failed: {}", e)
            return ""

        # Wait for processing
        while uploaded.state.name == "PROCESSING":
            time.sleep(0.5)
            uploaded = genai.get_file(uploaded.name)
        if uploaded.state.name == "FAILED":
            logger.error("Gemini video processing failed")
            return ""

        # Build prompt
        mic_section = (
            f"PLAYER SAID (mic transcript): \"{mic_transcript}\""
            if mic_transcript
            else "PLAYER SAID: (silent)"
        )
        prompt = (
            f"{mic_section}\n\n"
            "React in character as the genie. Maximum 2 sentences. "
            "If nothing notable happened, output exactly: SILENT"
        )

        try:
            response = self.model.generate_content(
                [uploaded, prompt],
                generation_config={
                    "max_output_tokens": self.cfg.llm["max_response_tokens"],
                    "temperature": self.cfg.llm["temperature"],
                },
            )
            text = (response.text or "").strip()
        except Exception as e:
            logger.error("Gemini call failed: {}", e)
            text = ""
        finally:
            # Clean up uploaded file
            try:
                genai.delete_file(uploaded.name)
            except Exception:
                pass

        if text.upper().startswith("SILENT"):
            return ""

        # Strip stage directions in case the model ignores instructions
        text = self._sanitize(text)
        return text

    @staticmethod
    def _sanitize(text: str) -> str:
        import re
        text = re.sub(r"\*[^*]*\*", "", text)  # *laughs*
        text = re.sub(r"\([^)]*\)", "", text)   # (sigh)
        text = text.replace("\n", " ").strip()
        return text
