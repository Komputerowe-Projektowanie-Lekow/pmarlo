from __future__ import annotations

import numpy as np
import pytest

mlc = pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


def _make_ds_two_contiguous_shards_same_traj():
    # Build two shards that are contiguous slices from the same trajectory
    X1 = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float64)
    X2 = np.random.default_rng(1).normal(size=(5, 3)).astype(np.float64)
    X = np.vstack([X1, X2])
    shards = [
        {
            "id": "s0",
            "start": 0,
            "stop": 5,
            "bias_potential": None,
            "temperature": 300.0,
            "source": {"traj": "a.dcd"},
        },
        {
            "id": "s1",
            "start": 5,
            "stop": 10,
            "bias_potential": None,
            "temperature": 300.0,
            "source": {"traj": "a.dcd"},
        },
    ]
    return {
        "X": X,
        "cv_names": ("a", "b", "c"),
        "periodic": (False, False, False),
        "__shards__": shards,
    }


def test_cross_shard_pairing_increases_pairs():
    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    ds = _make_ds_two_contiguous_shards_same_traj()
    # High lag such that within-shard yields few pairs, cross-boundary adds more
    lag = 3
    plan0 = TransformPlan(
        steps=(TransformStep("LEARN_CV", {"method": "deeptica", "lag": lag}),)
    )
    plan1 = TransformPlan(
        steps=(
            TransformStep(
                "LEARN_CV",
                {
                    "method": "deeptica",
                    "lag": lag,
                    "cross_shard_pairing": True,
                    "max_epochs": 1,
                    "early_stopping": 1,
                    "hidden": (4, 4),
                },
            ),
        )
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=lag)
    opts = BuildOpts(seed=0, temperature=300.0)

    res0 = build_result(ds, opts=opts, plan=plan0, applied=applied)
    res1 = build_result(ds, opts=opts, plan=plan1, applied=applied)

    art0 = (res0.artifacts or {}).get("mlcv_deeptica")
    art1 = (res1.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art0, dict) and isinstance(art1, dict)
    assert int(art1.get("pairs_total", 0)) >= int(art0.get("pairs_total", 0))
