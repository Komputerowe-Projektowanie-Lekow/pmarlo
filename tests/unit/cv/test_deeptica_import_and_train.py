from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mlc = pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")

# Accept either modern lightning or legacy pytorch_lightning
try:  # pragma: no cover - environment dependent
    import lightning as _lightning  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    _lightning = pytest.importorskip("pytorch_lightning")


def test_deeptica_fit_and_artifact_metrics(tmp_path: Path):
    # Local import after skipping heavy deps above
    from pmarlo.engine.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.features.deeptica import DeepTICAConfig, train_deeptica
    from pmarlo.features.pairs import scaled_time_pairs
    from pmarlo.transform.plan import TransformPlan, TransformStep

    rng = np.random.default_rng(0)
    X = rng.normal(size=(128, 3)).astype(np.float64)
    i, j = scaled_time_pairs(len(X), None, tau_scaled=3.0)

    cfg = DeepTICAConfig(
        lag=3,
        n_out=2,
        hidden=(8, 8),
        max_epochs=1,
        early_stopping=1,
        batch_size=64,
        seed=1,
    )
    model = train_deeptica([X], (i, j), cfg, weights=None)

    # Basic sanity on transform
    Z = model.transform(X)
    assert Z.shape == (X.shape[0], cfg.n_out)

    # Training telemetry present
    hist = getattr(model, "training_history", {})
    assert isinstance(hist, dict)
    loss = hist.get("loss_curve")
    assert loss is not None and isinstance(loss, list) and len(loss) >= 1

    # Engine integration: build with LEARN_CV and verify artifact metrics captured
    dataset = {"X": X, "cv_names": ("a", "b", "c"), "periodic": (False, False, False)}
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
                },
            ),
        )
    )
    applied = AppliedOpts(bins={"cv1": 8, "cv2": 8}, lag=3)
    opts = BuildOpts(seed=7, temperature=300.0)

    res = build_result(dataset, opts=opts, plan=plan, applied=applied)
    art = (res.artifacts or {}).get("mlcv_deeptica")
    assert isinstance(art, dict)
    assert art.get("applied") is True
    assert art.get("skipped") is False
    # Loss curve or proxy objective populated
    assert art.get("loss_curve") is not None
    # Environment probe present
    env = art.get("env")
    assert isinstance(env, dict) and "python_exe" in env
    # Pair stats present
    assert int(art.get("pairs_total", -1)) > 0
    assert int(art.get("lag_used", -1)) == 3
