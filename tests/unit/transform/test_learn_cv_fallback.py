from __future__ import annotations

import numpy as np
import pytest

mlc = pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


def _make_dataset_with_shards(lengths: list[int], k: int = 3, T: float = 300.0):
    n = int(sum(int(x) for x in lengths))
    rng = np.random.default_rng(5)
    X = rng.normal(size=(n, k)).astype(np.float64)
    shards = []
    offset = 0
    for i, m in enumerate(lengths):
        start = offset
        stop = offset + int(m)
        shards.append(
            {
                "id": f"shard_{i:04d}",
                "start": start,
                "stop": stop,
                "dtraj": None,
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


def test_lag_fallback_enables_training_when_smaller_lag_works():
    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    # Shards are too short for lag=5 but are sufficient for lag=2
    ds = _make_dataset_with_shards([3, 3], k=3)
    plan = TransformPlan(
        steps=(
            TransformStep(
                "LEARN_CV",
                {
                    "method": "deeptica",
                    "lag": 5,
                    "lag_fallback": [5, 4, 3, 2, 1],
                    "n_out": 2,
                    "hidden": (8, 8),
                    "max_epochs": 1,
                    "early_stopping": 1,
                    "batch_size": 16,
                },
            ),
        )
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=2)
    opts = BuildOpts(seed=11, temperature=300.0)

    res = build_result(ds, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    assert art.get("applied") is True
    assert art.get("reason") == "ok"
    # Confirm fallback used a lag <= 2
    assert int(art.get("lag_used", 999)) <= 2
    # Ladder recorded and attempts present
    assert isinstance(art.get("lag_fallback"), list)
    at = art.get("attempts")
    assert isinstance(at, list) and len(at) >= 2
