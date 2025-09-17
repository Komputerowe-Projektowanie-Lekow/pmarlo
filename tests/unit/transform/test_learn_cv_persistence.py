from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

mlc = pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


def test_learned_model_files_persist_when_applied(tmp_path: Path):
    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    rng = np.random.default_rng(42)
    # Single shard with ample frames
    X = rng.normal(size=(96, 3)).astype(np.float64)
    dataset = {
        "X": X,
        "cv_names": ("a", "b", "c"),
        "periodic": (False, False, False),
        "__shards__": [
            {
                "id": "shard_0000",
                "start": 0,
                "stop": int(X.shape[0]),
                "bias_potential": None,
                "temperature": 300.0,
            }
        ],
    }

    # Direct artifacts to a temp directory to avoid polluting user cache
    model_dir = tmp_path / "models"
    plan = TransformPlan(
        steps=(
            TransformStep(
                "LEARN_CV",
                {
                    "method": "deeptica",
                    "lag": 2,
                    "n_out": 2,
                    "hidden": (8, 8),
                    "max_epochs": 1,
                    "early_stopping": 1,
                },
            ),
        )
    )
    applied = AppliedOpts(
        bins={"cv1": 8, "cv2": 8}, lag=2, notes={"model_dir": str(model_dir)}
    )
    opts = BuildOpts(seed=123, temperature=300.0)

    # Ensure persistence is enabled
    os.environ.pop("PMARLO_PERSIST_MLCV", None)

    res = build_result(dataset, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    assert art.get("applied") is True
    files = art.get("files")
    assert isinstance(files, list) and len(files) > 0
    # Must include at least the core files
    joined = ",".join(files)
    assert ".json" in joined and ".pt" in joined and "scaler.pt" in joined


def test_no_files_when_skipped(tmp_path: Path):
    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    # Two too-short shards -> no pairs at lag=5
    ds = {
        "X": np.zeros((4 + 4, 3), dtype=np.float64),
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
    # Ensure model file list is absent or empty
    assert ("files" not in art) or (not art.get("files"))
