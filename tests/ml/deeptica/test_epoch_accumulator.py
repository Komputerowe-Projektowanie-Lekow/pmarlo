from __future__ import annotations

import numpy as np
import pytest

from pmarlo.ml.deeptica.trainer import _BatchOutcome, _EpochAccumulator


def test_epoch_accumulator_accepts_numpy_metrics():
    accumulator = _EpochAccumulator()
    metrics = {
        "cond_C00": 2.0,
        "cond_Ctt": 3.0,
        "var_z0": np.array([1.0, 2.0], dtype=np.float32),
        "var_zt": [0.5, 0.75],
        "mean_z0": np.array([0.1, -0.2], dtype=np.float32),
        "mean_zt": [0.05, -0.15],
        "eig_C00_min": 0.25,
        "eig_C00_max": 1.5,
        "eig_Ctt_min": 0.4,
        "eig_Ctt_max": 1.2,
    }

    outcome = _BatchOutcome(
        loss=0.5,
        score=0.75,
        batch_size=8,
        grad_norm=1.2,
        grad_norm_preclip=1.5,
        metrics=metrics,
    )

    accumulator.update(outcome)
    result = accumulator.finalize()

    assert np.allclose(result["var_z0"], [1.0, 2.0])
    assert np.allclose(result["var_zt"], [0.5, 0.75])
    assert np.allclose(result["mean_z0"], [0.1, -0.2])
    assert np.allclose(result["mean_zt"], [0.05, -0.15])
    assert result["cond_c00"] == pytest.approx(2.0)
    assert result["cond_ctt"] == pytest.approx(3.0)
    assert result["eig_c00_min"] == pytest.approx(0.25)
    assert result["eig_c00_max"] == pytest.approx(1.5)
    assert result["eig_ctt_min"] == pytest.approx(0.4)
    assert result["eig_ctt_max"] == pytest.approx(1.2)
