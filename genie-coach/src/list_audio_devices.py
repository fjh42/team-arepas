"""List audio input devices to help configure settings.yaml.
Run: python -m src.list_audio_devices
"""
import sounddevice as sd

if __name__ == "__main__":
    print(sd.query_devices())
    print("\nDefault input device:", sd.default.device)
