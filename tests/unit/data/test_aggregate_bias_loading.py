from __future__ import annotations

from pathlib import Path

import numpy as np

from pmarlo.data.aggregate import _maybe_read_bias


def test_maybe_read_bias_missing_file_returns_none(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.npz"
    assert _maybe_read_bias(missing_path) is None


def test_maybe_read_bias_reads_bias_array(tmp_path: Path) -> None:
    bias = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    np.savez(tmp_path / "with_bias.npz", bias_potential=bias)

    loaded = _maybe_read_bias(tmp_path / "with_bias.npz")
    assert loaded is not None
    np.testing.assert_allclose(loaded, bias)
