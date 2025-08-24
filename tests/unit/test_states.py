from __future__ import annotations

import numpy as np
import pytest

from pmarlo.states import (
    deserialize_macro_mapping,
    pcca_like_macrostates,
    serialize_macro_mapping,
)


def _example_T() -> np.ndarray:
    """Small transition matrix with two metastable basins."""
    return np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0],
            [0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ],
        dtype=float,
    )


def test_pcca_label_stability() -> None:
    pytest.importorskip("sklearn")
    T = _example_T()
    labels1 = pcca_like_macrostates(T, n_macrostates=2)
    labels2 = pcca_like_macrostates(T, n_macrostates=2)
    assert labels1 is not None and labels2 is not None
    assert np.array_equal(labels1, labels2)
    assert set(labels1.tolist()) == {0, 1}


def test_serialization_roundtrip() -> None:
    labels = np.array([0, 1, 1, 0], dtype=int)
    blob = serialize_macro_mapping(labels)
    restored = deserialize_macro_mapping(blob)
    assert np.array_equal(labels, restored)
