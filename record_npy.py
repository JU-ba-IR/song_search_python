from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from fingerprint import SAMPLE_RATE, create_fingerprint


BASE_DIR = Path(__file__).resolve().parent
MIC_CONFIG_FILE = BASE_DIR / "mic_config.json"
QUERY_WAV_FILE = BASE_DIR / "query.wav"
QUERY_FINGERPRINT_FILE = BASE_DIR / "query.npy"

DURATION_SECONDS = 12
CHANNELS = 1


def load_selected_microphone() -> int | None:
    if not MIC_CONFIG_FILE.exists():
        return None

    try:
        with MIC_CONFIG_FILE.open("r", encoding="utf-8") as file:
            config = json.load(file)

        device = config.get("device")
        return int(device) if device is not None else None

    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def main() -> None:
    device = load_selected_microphone()

    print(f"Using microphone: {device if device is not None else 'system default'}")
    print(f"Recording for {DURATION_SECONDS} seconds...")

    try:
        recording = sd.rec(
            int(DURATION_SECONDS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=device,
        )
        sd.wait()

    except Exception as error:
        raise RuntimeError(f"Microphone recording failed: {error}") from error

    recording = np.asarray(recording, dtype=np.float32).reshape(-1)

    peak = float(np.max(np.abs(recording)))
    rms = float(np.sqrt(np.mean(np.square(recording))))

    print("\nRecording quality")
    print("-----------------")
    print(f"Peak: {peak:.6f}")
    print(f"RMS:  {rms:.6f}")

    if peak >= 0.99:
        print("Warning: recording may be clipping. Lower the speaker or mic level.")

    if rms < 0.003:
        print("Warning: recording is very quiet.")
    elif rms < 0.015:
        print("Quality: usable")
    else:
        print("Quality: good")

    sf.write(
        QUERY_WAV_FILE,
        recording,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    fingerprint = create_fingerprint(recording, SAMPLE_RATE)

    if fingerprint.shape[0] == 0:
        raise RuntimeError(
            "No usable fingerprint was generated. "
            "Play the song louder or move the microphone closer."
        )

    np.save(QUERY_FINGERPRINT_FILE, fingerprint)

    print(f"\nSaved: {QUERY_WAV_FILE.name}")
    print(f"Saved: {QUERY_FINGERPRINT_FILE.name}")
    print(f"Query hashes: {fingerprint.shape[0]:,}")


if __name__ == "__main__":
    main()
