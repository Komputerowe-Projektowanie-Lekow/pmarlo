from __future__ import annotations

"""
Reproducible build demo: constructs a tiny dataset, builds MSM + FES
with explicit provenance (plan, options, dataset hash, digest), and
persists to JSON. Demonstrates roundtrip loading and validation.
"""

from pathlib import Path

import numpy as np

from pmarlo.engine import AppliedOpts, BuildOpts, build_result
from pmarlo.transform.plan import TransformPlan, TransformStep


def main() -> None:
    rng = np.random.default_rng(123)
    # Synthetic dataset: 2D CVs and simple 2-state dtrajs
    X = rng.normal(size=(400, 2))
    dtrajs = [
        np.array([0, 1] * 50 + [1, 0] * 50, dtype=int),
        np.array([1, 0] * 50 + [0, 1] * 50, dtype=int),
    ]
    dataset = {
        "X": X,
        "cv_names": ("phi", "psi"),
        "periodic": (False, False),
        "dtrajs": dtrajs,
    }

    # Plan & options
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 0.6}),))
    opts = BuildOpts(seed=123, n_states=2, temperature=300.0)
    applied = AppliedOpts(
        bins={"phi": 16, "psi": 16},
        lag=4,
        macrostates=3,
        notes={"fes": {"smooth": True}},
    )

    # Build and show provenance
    result = build_result(dataset, opts=opts, plan=plan, applied=applied)
    print("digest:", result.metadata.digest)
    print("dataset_hash:", result.metadata.dataset_hash)
    print("flags:", result.flags)

    # Persist and reload
    outdir = Path(__file__).parent / "programs_outputs" / "reproducible_build"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "reproducible_build.json"
    outpath.write_text(result.to_json())
    print("Saved:", outpath)

    from pmarlo.engine.build import BuildResult

    loaded = BuildResult.from_json(outpath.read_text())

    # Quick checks
    if loaded.stationary_distribution is not None:
        s = float(np.sum(loaded.stationary_distribution))
        assert abs(s - 1.0) < 1e-6
    assert loaded.fes is not None
    print("Roundtrip OK. MSM/FES present.")


if __name__ == "__main__":
    main()
