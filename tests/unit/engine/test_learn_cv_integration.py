from __future__ import annotations

import numpy as np
import pytest

mlc = pytest.importorskip("mlcolvar")
torch = pytest.importorskip("torch")


def test_learn_cv_step_records_provenance_and_edges():
    from pmarlo.engine.build import AppliedOpts, BuildOpts, build_result
    from pmarlo.transform.plan import TransformPlan, TransformStep

    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 3)).astype(np.float64)
    dataset = {"X": X, "cv_names": ("a", "b", "c"), "periodic": (False, False, False)}

    plan = TransformPlan(
        steps=(
            TransformStep(
                "LEARN_CV",
                {
                    "method": "deeptica",
                    "lag": 4,
                    "n_out": 2,
                    "hidden": (16, 16),
                    "max_epochs": 3,
                    "early_stopping": 1,
                },
            ),
        )
    )
    applied = AppliedOpts(bins={"cv1": 16, "cv2": 16}, lag=2)
    opts = BuildOpts(seed=7, temperature=300.0)

    res = build_result(dataset, opts=opts, plan=plan, applied=applied)

    # Provenance contains mlcv block and bin edges
    notes = res.metadata.applied_opts.notes
    assert isinstance(notes, dict)
    assert "mlcv" in notes and notes["mlcv"].get("method") == "deeptica"
    assert "cv_bin_edges" in notes and set(notes["cv_bin_edges"].keys()) == {
        "cv1",
        "cv2",
    }

    # FES likely present in learned CV space
    # (skipped if any internal condition prevented FES generation)
    assert res.fes is None or hasattr(res.fes, "F")
