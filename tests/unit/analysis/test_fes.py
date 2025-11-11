"""Tests for histogram smoothing in the FES helpers."""

from __future__ import annotations

import numpy as np

from pmarlo.analysis.fes import _smooth_sparse_bins


def _manual_smooth_sparse_bins(
    hist: np.ndarray, min_count: int
) -> tuple[np.ndarray, int]:
    mask = hist < float(min_count)
    if not np.any(mask):
        return hist.copy(), 0

    smoothed = hist.copy()
    padded = np.pad(hist, 1, mode="edge")
    coords = np.argwhere(mask)
    smoothed_bins = 0
    for i, j in coords:
        patch = padded[i : i + 3, j : j + 3]
        neighborhood = patch.reshape(-1)
        center_idx = neighborhood.size // 2
        neighborhood = np.delete(neighborhood, center_idx)
        neighbor_mean = float(np.mean(neighborhood)) if neighborhood.size else 0.0
        if neighbor_mean <= 0.0:
            continue
        target = max(neighbor_mean, float(min_count))
        if target > smoothed[i, j]:
            smoothed[i, j] = target
            smoothed_bins += 1
    return smoothed, smoothed_bins


def test_smooth_sparse_bins_matches_manual_implementation() -> None:
    histograms = [
        np.array(
            [
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 5.0, 0.2, 0.0],
                [0.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [1.0, 0.0, 3.0],
                [0.0, 2.5, 0.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ]

    for hist in histograms:
        for min_count in (1, 2):
            expected, expected_bins = _manual_smooth_sparse_bins(hist, min_count)
            result, result_bins = _smooth_sparse_bins(hist, min_count)

            np.testing.assert_allclose(result, expected)
            assert result_bins == expected_bins


def test_smooth_sparse_bins_keeps_populated_bins() -> None:
    hist = np.array(
        [
            [3.0, 2.1],
            [5.0, 1.6],
        ],
        dtype=np.float64,
    )

    result, smoothed_bins = _smooth_sparse_bins(hist, min_count=1)

    np.testing.assert_allclose(result, hist)
    assert smoothed_bins == 0
