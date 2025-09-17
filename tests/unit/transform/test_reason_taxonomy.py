from __future__ import annotations

import types

import numpy as np


def _tiny_dataset(n: int = 8, k: int = 3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, k)).astype(np.float64)
    return {
        "X": X,
        "cv_names": tuple(f"f{i}" for i in range(k)),
        "periodic": tuple(False for _ in range(k)),
    }


def test_reason_missing_dependency(monkeypatch):
    # Arrange: force the env probe to report missing lightning modules
    import importlib

    real_import_module = importlib.import_module

    def fake_import(name, package=None):  # noqa: ANN001
        if name in {"lightning", "pytorch_lightning"}:
            raise ImportError(f"No module named '{name}'")
        return real_import_module(name, package=package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    ds = _tiny_dataset(n=32, k=3)
    plan = TransformPlan(
        steps=(TransformStep("LEARN_CV", {"method": "deeptica", "lag": 3}),)
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=3)
    opts = BuildOpts(seed=0, temperature=300.0)

    res = build_result(ds, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    assert art.get("skipped") is True
    assert str(art.get("reason", "")).startswith("missing_dependency:")
    # env and missing list present
    env = art.get("env")
    assert isinstance(env, dict)
    missing = art.get("missing")
    assert isinstance(missing, list) and (
        "lightning" in missing or "pytorch_lightning" in missing
    )
    # Pair stats recorded despite missing deps
    assert "pairs_total" in art and "per_shard" in art


def test_reason_api_incompatibility(monkeypatch):
    # Arrange: force training loader to raise a class named PmarloApiIncompatibilityError
    from pmarlo.transform import build as build_mod

    PAI = types.new_class("PmarloApiIncompatibilityError", (RuntimeError,))

    def boom(*args, **kwargs):  # noqa: ANN001, ANN003
        raise PAI("API mismatch")

    monkeypatch.setattr(build_mod, "_load_or_train_model", boom)

    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    ds = _tiny_dataset(n=32, k=3)
    plan = TransformPlan(
        steps=(TransformStep("LEARN_CV", {"method": "deeptica", "lag": 3}),)
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=3)
    opts = BuildOpts(seed=0, temperature=300.0)

    res = build_result(ds, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    assert art.get("skipped") is True
    assert art.get("reason") == "api_incompatibility"
    assert isinstance(art.get("traceback"), str) and "API mismatch" in art.get(
        "traceback", ""
    )


def test_reason_no_pairs():
    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    # Two short shards with lag too large to form any pairs
    X = np.zeros((8, 3), dtype=np.float64)
    ds = {
        "X": X,
        "cv_names": ("a", "b", "c"),
        "periodic": (False, False, False),
        "__shards__": [
            {
                "id": "s0",
                "start": 0,
                "stop": 4,
                "bias_potential": None,
                "temperature": 300.0,
            },
            {
                "id": "s1",
                "start": 4,
                "stop": 8,
                "bias_potential": None,
                "temperature": 300.0,
            },
        ],
    }
    plan = TransformPlan(
        steps=(TransformStep("LEARN_CV", {"method": "deeptica", "lag": 5}),)
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=5)
    opts = BuildOpts(seed=0, temperature=300.0)

    res = build_result(ds, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    assert art.get("skipped") is True
    assert art.get("reason") == "no_pairs"
    per = art.get("per_shard")
    assert isinstance(per, list) and len(per) == 2


def test_reason_exception(monkeypatch):
    # Arrange: force training loader to raise a generic error
    from pmarlo.transform import build as build_mod

    def boom(*args, **kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("boom")

    monkeypatch.setattr(build_mod, "_load_or_train_model", boom)

    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    ds = _tiny_dataset(n=32, k=3)
    plan = TransformPlan(
        steps=(TransformStep("LEARN_CV", {"method": "deeptica", "lag": 3}),)
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=3)
    opts = BuildOpts(seed=0, temperature=300.0)

    res = build_result(ds, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    assert art.get("skipped") is True
    assert art.get("reason") == "exception"
    assert isinstance(art.get("traceback"), str) and "boom" in art.get("traceback", "")
