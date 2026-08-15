from __future__ import annotations

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
from scipy.ndimage import maximum_filter


SAMPLE_RATE = 22_050
N_FFT = 4_096
HOP_LENGTH = 512

MIN_FREQUENCY_HZ = 180
MAX_FREQUENCY_HZ = 5_500
PEAK_THRESHOLD_DB = -35.0

# Quantisation makes microphone recordings more tolerant to small
# frequency and timing differences.
FREQUENCY_QUANTISATION_BINS = 2
TIME_QUANTISATION_FRAMES = 2

FAN_VALUE = 12
MIN_PAIR_DELTA_SECONDS = 0.08
MAX_PAIR_DELTA_SECONDS = 3.0


def load_audio(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load an audio file as mono audio using the system sample rate."""
    audio, sample_rate = librosa.load(
        str(path),
        sr=SAMPLE_RATE,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(f"No audio samples were loaded from: {path}")

    audio = np.asarray(audio, dtype=np.float32)
    audio = audio - float(np.mean(audio))

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        raise ValueError(f"The audio is silent or almost silent: {path}")

    audio = audio / peak
    return audio, sample_rate


def create_fingerprint(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Create a fingerprint array with columns:

    [quantised_frequency_1, quantised_frequency_2,
     quantised_time_delta, quantised_anchor_time]
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size == 0:
        return np.empty((0, 4), dtype=np.int32)

    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))

    if peak < 1e-8:
        return np.empty((0, 4), dtype=np.int32)

    audio = audio / peak

    stft = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window="hann",
        center=True,
    )

    db = librosa.amplitude_to_db(
        np.abs(stft),
        ref=np.max,
        top_db=80.0,
    )

    local_maxima = db == maximum_filter(
        db,
        size=(15, 9),
        mode="constant",
    )

    frequency_bins, time_frames = np.where(
        local_maxima & (db >= PEAK_THRESHOLD_DB)
    )

    frequencies_hz = librosa.fft_frequencies(
        sr=sample_rate,
        n_fft=N_FFT,
    )[frequency_bins]

    valid = (
        (frequencies_hz >= MIN_FREQUENCY_HZ)
        & (frequencies_hz <= MAX_FREQUENCY_HZ)
    )

    frequency_bins = frequency_bins[valid]
    time_frames = time_frames[valid]

    if frequency_bins.size < 2:
        return np.empty((0, 4), dtype=np.int32)

    # Sort first by time and then by frequency for deterministic output.
    order = np.lexsort((frequency_bins, time_frames))
    frequency_bins = frequency_bins[order]
    time_frames = time_frames[order]

    min_delta_frames = max(
        1,
        int(round(MIN_PAIR_DELTA_SECONDS * sample_rate / HOP_LENGTH)),
    )
    max_delta_frames = int(
        round(MAX_PAIR_DELTA_SECONDS * sample_rate / HOP_LENGTH)
    )

    hashes: list[tuple[int, int, int, int]] = []

    for anchor_index in range(len(time_frames)):
        anchor_time = int(time_frames[anchor_index])
        anchor_frequency = int(frequency_bins[anchor_index])

        added = 0

        for target_index in range(anchor_index + 1, len(time_frames)):
            target_time = int(time_frames[target_index])
            delta_frames = target_time - anchor_time

            if delta_frames < min_delta_frames:
                continue

            if delta_frames > max_delta_frames:
                break

            target_frequency = int(frequency_bins[target_index])

            hashes.append(
                (
                    anchor_frequency // FREQUENCY_QUANTISATION_BINS,
                    target_frequency // FREQUENCY_QUANTISATION_BINS,
                    delta_frames // TIME_QUANTISATION_FRAMES,
                    anchor_time // TIME_QUANTISATION_FRAMES,
                )
            )

            added += 1
            if added >= FAN_VALUE:
                break

    if not hashes:
        return np.empty((0, 4), dtype=np.int32)

    fingerprint = np.asarray(hashes, dtype=np.int32)

    # Remove duplicate hashes without changing the four-column format.
    fingerprint = np.unique(fingerprint, axis=0)

    return fingerprint


def fingerprint_file(path: str | Path) -> np.ndarray:
    """Load an audio file and create its fingerprint."""
    audio, sample_rate = load_audio(path)
    return create_fingerprint(audio, sample_rate)
