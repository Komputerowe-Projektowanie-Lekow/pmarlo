from __future__ import annotations

import numpy as np

from pmarlo.features.diagnostics import diagnose_deeptica_pairs


def _dataset_with_shards(lengths, k=3, T=300.0):  # noqa: ANN001
    n = int(sum(int(x) for x in lengths))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, k)).astype(np.float64)
    shards = []
    offset = 0
    for i, m in enumerate(lengths):
        start = offset
        stop = offset + int(m)
        shards.append(
            {
                "id": f"shard_{i:02d}",
                "frames": int(m),
                "bias_potential": None,
                "temperature": float(T),
            }
        )
        offset = stop
    return {
        "X": X,
        "cv_names": ("a", "b", "c"),
        "periodic": (False, False, False),
        "__shards__": shards,
    }


def test_diagnostics_reports_no_pairs_when_lag_too_large():
    ds = _dataset_with_shards([4, 4], k=3)
    rep = diagnose_deeptica_pairs(ds, lag=5)
    assert rep.pairs_total == 0
    assert rep.n_shards == 2
    assert "too short for lag" in rep.message


def test_diagnostics_reports_pairs_for_reasonable_lag():
    ds = _dataset_with_shards([12, 20], k=3)
    rep = diagnose_deeptica_pairs(ds, lag=3)
    assert rep.pairs_total > 0
    assert rep.n_shards == 2
    assert "Uniform pairs:" in rep.message
