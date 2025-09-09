from __future__ import annotations

import numpy as np
import pytest

mlc = pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


def _dataset_mixed():
    # One short shard (no pairs at lag=3), one longer shard (produces pairs)
    X1 = np.random.default_rng(0).normal(size=(2, 3)).astype(np.float64)
    X2 = np.random.default_rng(1).normal(size=(12, 3)).astype(np.float64)
    X = np.vstack([X1, X2])
    shards = [
        {
            "id": "s0",
            "start": 0,
            "stop": 2,
            "bias_potential": None,
            "temperature": 300.0,
        },
        {
            "id": "s1",
            "start": 2,
            "stop": 14,
            "bias_potential": None,
            "temperature": 300.0,
        },
    ]
    return {
        "X": X,
        "cv_names": ("a", "b", "c"),
        "periodic": (False, False, False),
        "__shards__": shards,
    }


def test_mixed_shards_some_pairs_total_positive():
    from pmarlo.engine.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    ds = _dataset_mixed()
    plan = TransformPlan(
        steps=(
            TransformStep(
                "LEARN_CV",
                {
                    "method": "deeptica",
                    "lag": 3,
                    "max_epochs": 1,
                    "early_stopping": 1,
                    "hidden": (8, 8),
                },
            ),
        )
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=3)
    opts = BuildOpts(seed=2, temperature=300.0)

    res = build_result(ds, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    # Either applied or at least not "no_pairs"
    assert int(art.get("pairs_total", 0)) > 0
