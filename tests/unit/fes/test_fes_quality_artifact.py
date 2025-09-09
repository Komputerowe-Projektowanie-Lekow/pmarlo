from __future__ import annotations

import numpy as np

from pmarlo.engine.build import AppliedOpts, BuildOpts, build_result
from pmarlo.transform.plan import TransformPlan, TransformStep


def test_fes_quality_metric_in_artifacts():
    # Construct sparse data by sampling from a tiny support relative to bins
    rng = np.random.default_rng(0)
    X = rng.normal(loc=0.0, scale=0.01, size=(200, 2)).astype(np.float64)
    dataset = {"X": X, "cv_names": ("x", "y"), "periodic": (False, False)}
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 0.6}),))
    applied = AppliedOpts(bins={"x": 64, "y": 64}, lag=2)
    opts = BuildOpts(seed=1, temperature=300.0)

    res = build_result(dataset, opts=opts, plan=plan, applied=applied)
    art = res.artifacts or {}
    fq = art.get("fes_quality")
    assert isinstance(fq, dict)
    frac = float(fq.get("empty_bins_fraction", 0.0))
    assert 0.0 <= frac <= 1.0
    # Likely sparse given narrow distribution
    assert fq.get("warn_sparse") in (True, False)
