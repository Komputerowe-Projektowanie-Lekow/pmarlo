from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlcolvar")
pytest.importorskip("torch")


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


def test_learn_cv_uses_requested_lag_only():
    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    ds = _make_dataset_with_shards([32, 32], k=3)
    plan = TransformPlan(
        steps=(
            TransformStep(
                "LEARN_CV",
                {
                    "method": "deeptica",
                    "lag": 3,
                    "n_out": 2,
                    "hidden": (8, 8),
                    "max_epochs": 1,
                    "early_stopping": 1,
                    "batch_size": 16,
                },
            ),
        )
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=3)
    opts = BuildOpts(seed=11, temperature=300.0)

    res = build_result(ds, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    if not art.get("applied", False):
        pytest.skip("DeepTICA extras unavailable")

    assert art.get("lag_used") == 3
    assert art.get("lag_candidates") == [3]
    assert "lag_fallback" not in art
    attempts = art.get("attempts")
    assert isinstance(attempts, list) and len(attempts) == 1
    assert attempts[0].get("lag") == 3


def test_collect_lag_candidates_rejects_non_integer_values():
    from pmarlo.transform import apply as apply_mod

    with pytest.raises(
        ValueError, match=r"LEARN_CV requires a positive integer lag value; params\['lag']='abc' is not an integer"
    ):
        apply_mod._collect_lag_candidates({"lag": "abc"}, 5)


def test_collect_lag_candidates_requires_positive_requested_lag():
    from pmarlo.transform import apply as apply_mod

    with pytest.raises(
        ValueError, match=r"LEARN_CV requires a positive integer lag value; requested lag=0 is not positive"
    ):
        apply_mod._collect_lag_candidates({}, 0)


def test_collect_lag_candidates_flags_missing_lag_entry():
    from pmarlo.transform import apply as apply_mod

    with pytest.raises(
        ValueError, match=r"LEARN_CV requires a positive integer lag value; params\['lag'] is missing"
    ):
        apply_mod._collect_lag_candidates({"lag": None}, 5)
