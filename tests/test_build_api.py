import numpy as np

from pmarlo.engine.build import AppliedOpts, BuildOpts, build_result
from pmarlo.transform.plan import TransformPlan, TransformStep


def test_build_result_metadata_and_msm_stationarity():
    # Tiny synthetic dataset with discrete trajectories
    dtrajs = [
        np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int),
        np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=int),
    ]
    dataset = {"dtrajs": dtrajs}

    # Simple transform plan
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 1.0}),))

    # Applied options reflect resolved optimization decisions
    applied = AppliedOpts(bins={"phi": 16, "psi": 16}, lag=4, macrostates=3)

    # Declarative knobs (not all must be used)
    opts = BuildOpts(seed=42, count_mode="sliding", n_states=2)

    result = build_result(dataset, opts=opts, plan=plan, applied=applied)

    # Metadata is attached and preserves provenance
    assert tuple(result.metadata.transform_plan) == plan.steps
    assert result.metadata.applied_opts.lag == 4
    assert result.metadata.applied_opts.bins == {"phi": 16, "psi": 16}

    # MSM stationary distribution, if present, sums to ~1
    if result.stationary_distribution is not None:
        s = float(np.sum(result.stationary_distribution))
        assert np.isfinite(s)
        assert abs(s - 1.0) < 1e-6


def test_build_result_with_fes_embeds_provenance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    dataset = {"X": X, "cv_names": ("phi", "psi"), "periodic": (False, False)}

    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 0.5}),))

    applied = AppliedOpts(
        bins={"phi": 16, "psi": 16}, lag=2, notes={"fes": {"smooth": True}}
    )
    opts = BuildOpts(seed=123, temperature=300.0)

    result = build_result(dataset, opts=opts, plan=plan, applied=applied)

    assert result.fes is not None
    # FES provenance must be tracked in metadata (structured block)
    assert result.metadata.fes is not None
    assert result.metadata.fes.get("bins") == (16, 16)
    assert tuple(result.metadata.fes.get("names", ())) == ("phi", "psi")


def test_build_result_json_roundtrip_preserves_shapes():
    dtrajs = [np.array([0, 1, 0, 1, 0, 1], dtype=int)]
    dataset = {"dtrajs": dtrajs}
    plan = TransformPlan(steps=())
    applied = AppliedOpts(lag=2)
    opts = BuildOpts(seed=7, n_states=2)
    result = build_result(dataset, opts=opts, plan=plan, applied=applied)
    text = result.to_json()
    from pmarlo.engine.build import BuildResult

    loaded = BuildResult.from_json(text)
    assert (loaded.transition_matrix is None) == (result.transition_matrix is None)
    if loaded.transition_matrix is not None and result.transition_matrix is not None:
        assert loaded.transition_matrix.shape == result.transition_matrix.shape
