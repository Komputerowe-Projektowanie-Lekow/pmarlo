from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def load(debug_dir: str) -> tuple[dict, np.ndarray]:
    directory = Path(debug_dir)
    summary_path = directory / "summary.json"
    counts_path = directory / "transition_counts.npy"

    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    summary = payload.get("summary", payload)
    counts = np.load(counts_path)
    return summary, counts


def diag_mass(counts: np.ndarray) -> float:
    if counts.size == 0 or counts.shape[0] == 0:
        return float("nan")
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    T = counts / row_sum
    return float(np.trace(T) / T.shape[0])


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_debug_bundles.py <debug_A> <debug_B>")
        return 1

    a_dir, b_dir = sys.argv[1:]
    if not os.path.isdir(a_dir) or not os.path.isdir(b_dir):
        print("Both arguments must be directories containing debug bundles.")
        return 1

    summary_a, counts_a = load(a_dir)
    summary_b, counts_b = load(b_dir)

    print("A:", a_dir, json.dumps(summary_a, indent=2))
    print("B:", b_dir, json.dumps(summary_b, indent=2))

    dm_a = diag_mass(counts_a)
    dm_b = diag_mass(counts_b)

    print("Δdiag_mass:", dm_b - dm_a)
    print("Counts equal?:", np.array_equal(counts_a, counts_b))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
