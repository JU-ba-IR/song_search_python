from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
FINGERPRINT_FOLDER = BASE_DIR / "fingerprints"
QUERY_FILE = BASE_DIR / "query.npy"

FREQUENCY_TOLERANCE = 1
DELTA_TIME_TOLERANCE = 1
OFFSET_CLUSTER_RADIUS = 1

MINIMUM_VOTES = 8
MINIMUM_TOP_TO_SECOND_RATIO = 1.25


def build_index(database: np.ndarray) -> dict[tuple[int, int, int], list[int]]:
    index: dict[tuple[int, int, int], list[int]] = defaultdict(list)

    for frequency_1, frequency_2, delta_time, anchor_time in database:
        key = (
            int(frequency_1),
            int(frequency_2),
            int(delta_time),
        )
        index[key].append(int(anchor_time))

    return dict(index)


def clustered_best_vote(offset_votes: Counter[int]) -> tuple[int, int]:
    if not offset_votes:
        return 0, 0

    best_offset = 0
    best_votes = 0

    for offset in offset_votes:
        cluster_votes = sum(
            offset_votes.get(offset + neighbour, 0)
            for neighbour in range(
                -OFFSET_CLUSTER_RADIUS,
                OFFSET_CLUSTER_RADIUS + 1,
            )
        )

        if cluster_votes > best_votes:
            best_votes = cluster_votes
            best_offset = offset

    return best_votes, best_offset


def match_song(
    query: np.ndarray,
    database: np.ndarray,
) -> tuple[int, float, int]:
    database_index = build_index(database)
    offset_votes: Counter[int] = Counter()
    matched_query_hashes = 0

    for query_row in query:
        query_f1, query_f2, query_delta, query_time = map(int, query_row)
        offsets_for_this_query_hash: set[int] = set()

        for f1_change in range(
            -FREQUENCY_TOLERANCE,
            FREQUENCY_TOLERANCE + 1,
        ):
            for f2_change in range(
                -FREQUENCY_TOLERANCE,
                FREQUENCY_TOLERANCE + 1,
            ):
                for delta_change in range(
                    -DELTA_TIME_TOLERANCE,
                    DELTA_TIME_TOLERANCE + 1,
                ):
                    key = (
                        query_f1 + f1_change,
                        query_f2 + f2_change,
                        query_delta + delta_change,
                    )

                    for database_time in database_index.get(key, ()):
                        offsets_for_this_query_hash.add(
                            database_time - query_time
                        )

        if offsets_for_this_query_hash:
            matched_query_hashes += 1
            offset_votes.update(offsets_for_this_query_hash)

    best_votes, best_offset = clustered_best_vote(offset_votes)

    coverage = (
        100.0 * matched_query_hashes / max(1, len(query))
    )

    return best_votes, coverage, best_offset


def main() -> None:
    if not QUERY_FILE.exists():
        print("query.npy was not found.")
        print("Record a sample first by running: python record_npy.py")
        return

    if not FINGERPRINT_FOLDER.exists():
        print("The fingerprints folder was not found.")
        print("Run: python create_database.py")
        return

    query = np.load(QUERY_FILE, allow_pickle=False)

    if query.ndim != 2 or query.shape[1] != 4 or len(query) == 0:
        print("query.npy is empty or has the wrong format.")
        print("Record the query again.")
        return

    fingerprint_files = sorted(FINGERPRINT_FOLDER.glob("*.npy"))

    if not fingerprint_files:
        print("No database fingerprints were found.")
        print("Run: python create_database.py")
        return

    print(f"Query hashes: {len(query):,}")
    print(f"Comparing against {len(fingerprint_files)} song(s)...\n")

    results: list[dict[str, object]] = []

    for fingerprint_file in fingerprint_files:
        try:
            database = np.load(fingerprint_file, allow_pickle=False)

            if (
                database.ndim != 2
                or database.shape[1] != 4
                or len(database) == 0
            ):
                print(f"Skipping invalid fingerprint: {fingerprint_file.name}")
                continue

            votes, coverage, offset = match_song(query, database)

            results.append(
                {
                    "name": fingerprint_file.stem,
                    "votes": votes,
                    "coverage": coverage,
                    "offset": offset,
                }
            )

        except Exception as error:
            print(f"Could not read {fingerprint_file.name}: {error}")

    results.sort(
        key=lambda result: (
            int(result["votes"]),
            float(result["coverage"]),
        ),
        reverse=True,
    )

    if not results:
        print("No valid fingerprints could be compared.")
        return

    print("===== Ranking =====\n")

    for rank, result in enumerate(results[:10], start=1):
        print(
            f"{rank}. {result['name']} | "
            f"aligned votes: {result['votes']} | "
            f"hash coverage: {result['coverage']:.2f}%"
        )

    best = results[0]
    second_votes = (
        int(results[1]["votes"])
        if len(results) > 1
        else 0
    )

    top_votes = int(best["votes"])
    ratio = (
        top_votes / max(1, second_votes)
        if second_votes > 0
        else float("inf")
    )

    print("\n===== Best Match =====\n")

    if top_votes < MINIMUM_VOTES:
        print("No reliable match.")
        print("Reason: too few time-aligned fingerprint votes.")
        return

    if len(results) > 1 and ratio < MINIMUM_TOP_TO_SECOND_RATIO:
        print("Match is uncertain.")
        print(
            f"Top candidates are too close "
            f"(separation ratio: {ratio:.2f}x)."
        )
        print(f"Most likely: {best['name']}")
        return

    print(best["name"])
    print(f"Aligned votes: {top_votes}")
    print(
        "Separation from second place: "
        + (
            f"{ratio:.2f}x"
            if np.isfinite(ratio)
            else "no competing match"
        )
    )


if __name__ == "__main__":
    main()
