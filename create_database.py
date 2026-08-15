from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from fingerprint import fingerprint_file


BASE_DIR = Path(__file__).resolve().parent
SONG_FOLDER = BASE_DIR / "songs"
FINGERPRINT_FOLDER = BASE_DIR / "fingerprints"
DATABASE_FILE = BASE_DIR / "database.json"

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
}


def main() -> None:
    SONG_FOLDER.mkdir(parents=True, exist_ok=True)
    FINGERPRINT_FOLDER.mkdir(parents=True, exist_ok=True)

    songs = sorted(
        path
        for path in SONG_FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not songs:
        print(f"No supported audio files were found in:\n{SONG_FOLDER}")
        print("Add WAV files to the songs folder and run this file again.")
        return

    database: list[dict[str, object]] = []
    successful = 0

    print(f"Found {len(songs)} song(s).\n")

    for number, song_path in enumerate(songs, start=1):
        print(f"[{number}/{len(songs)}] Processing: {song_path.name}")

        try:
            fingerprint = fingerprint_file(song_path)

            if fingerprint.shape[0] == 0:
                print("  Skipped: no usable fingerprint was generated.\n")
                continue

            save_path = FINGERPRINT_FOLDER / f"{song_path.stem}.npy"
            np.save(save_path, fingerprint)

            database.append(
                {
                    "name": song_path.stem,
                    "audio_file": os.path.relpath(song_path, BASE_DIR),
                    "fingerprint_file": os.path.relpath(save_path, BASE_DIR),
                    "hash_count": int(fingerprint.shape[0]),
                }
            )

            successful += 1
            print(f"  Saved: {save_path.name}")
            print(f"  Fingerprint hashes: {fingerprint.shape[0]:,}\n")

        except Exception as error:
            print(f"  Failed: {error}\n")

    with DATABASE_FILE.open("w", encoding="utf-8") as file:
        json.dump(database, file, indent=2, ensure_ascii=False)

    print("=" * 50)
    print(f"Completed: {successful}/{len(songs)} song(s)")
    print(f"Fingerprint folder: {FINGERPRINT_FOLDER}")
    print(f"Database file: {DATABASE_FILE}")


if __name__ == "__main__":
    main()
