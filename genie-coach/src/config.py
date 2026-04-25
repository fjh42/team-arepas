"""Configuration loader."""
from pathlib import Path
from typing import Any
import os
import yaml
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config:
    def __init__(self, settings_path: str = "config/settings.yaml"):
        full_path = PROJECT_ROOT / settings_path
        with open(full_path, "r") as f:
            self._data = yaml.safe_load(f)

        # Load persona
        persona_path = PROJECT_ROOT / self._data["llm"]["persona_file"]
        with open(persona_path, "r") as f:
            self.persona = f.read()

        # API keys from env
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "")

        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY missing in .env")

    def __getattr__(self, key: str) -> Any:
        if key in self._data:
            return self._data[key]
        raise AttributeError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


# Singleton
_config_instance: Config | None = None


def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
