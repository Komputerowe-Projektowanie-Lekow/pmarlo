from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
from pmarlo.transform.plan import TransformPlan, TransformStep


def _make_dataset_with_shards(lengths: list[int], k: int = 3, T: float = 300.0):
    """Construct a synthetic dataset dict with __shards__ metadata.

    Each length in `lengths` produces a shard of that many frames; feature
    values are random but reproducible.
    """
    n = int(sum(int(x) for x in lengths))
    rng = np.random.default_rng(123)
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


def test_deeptica_artifact_no_pairs_when_shards_short():
    # Two short shards; lag requires pairs but none possible
    dataset = _make_dataset_with_shards([4, 4], k=3)
    plan = TransformPlan(
        steps=(TransformStep("LEARN_CV", {"method": "deeptica", "lag": 5}),)
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=5)
    opts = BuildOpts(seed=7, temperature=300.0)

    res = build_result(dataset, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    assert art.get("skipped") is True
    assert art.get("reason") == "no_pairs"
    assert int(art.get("n_shards", -1)) == 2
    assert int(art.get("frames_total", -1)) == 8
    assert int(art.get("pairs_total", -1)) == 0
    # Per-shard reporting present
    per = art.get("per_shard")
    assert isinstance(per, list) and len(per) == 2
    assert all("frames" in d and "pairs" in d for d in per)
    # Warnings should be present (too-short shards and low total frames)
    warns = art.get("warnings")
    assert isinstance(warns, list) and len(warns) >= 1


def test_deeptica_artifact_exception_is_recorded(monkeypatch):
    # Long enough shard to produce pairs; then force training exception
    dataset = _make_dataset_with_shards([32], k=3)
    plan = TransformPlan(
        steps=(TransformStep("LEARN_CV", {"method": "deeptica", "lag": 3}),)
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=3)
    opts = BuildOpts(seed=7, temperature=300.0)

    # Force _load_or_train_model to raise
    import pmarlo.transform.build as eng

    def boom(*args, **kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("boom")

    monkeypatch.setattr(eng, "_load_or_train_model", boom)

    res = build_result(dataset, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    assert art.get("skipped") is True
    assert art.get("reason") == "exception"
    assert "boom" in str(art.get("error", ""))
