from __future__ import annotations

import numpy as np
import pytest

from pmarlo.markov_state_model.picker import find_local_minima_2d


def _reference_minima(array: np.ndarray) -> list[tuple[int, int]]:
    minima: list[tuple[int, int]] = []
    nx, ny = array.shape
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            val = array[i, j]
            if not np.isfinite(val):
                continue
            neighbors = array[i - 1 : i + 2, j - 1 : j + 2]
            if np.all(val <= neighbors) and np.any(val < neighbors):
                minima.append((i, j))
    return minima


def test_find_local_minima_matches_reference() -> None:
    array = np.array(
        [
            [5.0, 4.0, 6.0, 7.0, 8.0],
            [4.0, 1.0, 5.0, 6.0, 7.0],
            [6.0, 5.0, 9.0, 3.0, 6.0],
            [7.0, 6.0, 4.0, 5.0, 7.0],
            [8.0, 7.0, 6.0, 7.0, 8.0],
        ],
        dtype=float,
    )

    expected = _reference_minima(array)
    result = find_local_minima_2d(array)

    assert sorted(result) == sorted(expected)


def test_find_local_minima_handles_invalid_values() -> None:
    array = np.array(
        [
            [5.0, 4.0, 6.0, 7.0],
            [4.0, 1.0, 5.0, np.nan],
            [6.0, 5.0, 9.0, 3.0],
            [np.inf, 6.0, 4.0, 5.0],
        ],
        dtype=float,
    )

    expected = _reference_minima(array)
    result = find_local_minima_2d(array)

    assert result == expected


def test_find_local_minima_requires_sufficient_size() -> None:
    small = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=float)
    assert find_local_minima_2d(small) == []


def test_find_local_minima_requires_2d_input() -> None:
    array = np.zeros((2, 2, 2), dtype=float)
    with pytest.raises(ValueError):
        find_local_minima_2d(array)
