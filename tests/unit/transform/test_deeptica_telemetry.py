from __future__ import annotations

import numpy as np
import pytest

mlc = pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


def test_training_telemetry_present_in_artifacts():
    from pmarlo.transform.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    rng = np.random.default_rng(123)
    X = rng.normal(size=(64, 3)).astype(np.float64)
    ds = {"X": X, "cv_names": ("a", "b", "c"), "periodic": (False, False, False)}
    plan = TransformPlan(
        steps=(
            TransformStep(
                "LEARN_CV",
                {
                    "method": "deeptica",
                    "lag": 2,
                    "max_epochs": 2,
                    "early_stopping": 1,
                    "hidden": (8, 8),
                },
            ),
        )
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=2)
    opts = BuildOpts(seed=7, temperature=300.0)

    res = build_result(ds, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    if art.get("applied"):
        # Curves recorded
        assert len(art.get("objective_curve", [])) >= 1
        assert len(art.get("loss_curve", [])) >= 1
        assert art.get("wall_time_s", 0.0) >= 0.0
        assert art.get("best_val_score") is not None
        assert art.get("best_epoch") is not None
        assert art.get("best_tau") is not None
        assert isinstance(art["best_val_score"], float)
        assert isinstance(art["best_epoch"], int)
        assert isinstance(art["best_tau"], int)
