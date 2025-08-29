from __future__ import annotations

"""
Tiny demo: emit shards from trajectory slices → aggregate → build → archive.

This script uses the bundled test trajectory and PDB to compute two simple
CVs per frame (radius of gyration and RMSD to the first frame), splits the
trajectory into a few slices (as shard inputs), emits deterministic shards,
aggregates them into a single dataset, builds MSM/FES via the provenance-first
engine, and saves a portable JSON bundle with a dataset hash and digest.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Set, Tuple

import numpy as np

from pmarlo import aggregate_and_build, emit_shards_from_trajectories
from pmarlo.engine import AppliedOpts, BuildOpts
from pmarlo.transform.plan import TransformPlan, TransformStep

BASE = Path(__file__).resolve().parents[1]
TESTS = BASE / "tests" / "data"
PDB = TESTS / "3gd8-fixed.pdb"
DCD = TESTS / "traj.dcd"

OUT_BASE = Path(__file__).resolve().parent / "programs_outputs" / "sharded_build"
SHARDS_DIR = OUT_BASE / "shards"
BUNDLE = OUT_BASE / "build.json"


def _make_slice_descriptors(n_frames: int, n_slices: int) -> list[Path]:
    """Create empty files encoding [start:stop) frame ranges in their names."""
    edges = np.linspace(0, n_frames, n_slices + 1, dtype=int)
    desc = []
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(n_slices):
        a, b = int(edges[i]), int(edges[i + 1])
        p = SHARDS_DIR / f"slice_{a:06d}_{b:06d}.dcd"
        p.write_bytes(b"")
        desc.append(p)
    return desc


def _make_extractor_closure(traj, bins: Dict[str, int]) -> Tuple[
    Callable[[Path], Tuple[Dict[str, np.ndarray], Optional[np.ndarray], Dict]],
    Set[str],
]:
    """Bind an extractor over an already-loaded trajectory (avoid repeated I/O).

    Also generates a simple discrete trajectory by 2D binning on (rg, rmsd)
    using the provided ``bins`` per CV name.
    """
    import re

    import mdtraj as md

    def _extract(path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray | None, Dict]:
        m = re.search(r"slice_(\d+)_(\d+)\.dcd$", path.name)
        if not m:
            raise ValueError(f"Unexpected slice filename: {path}")
        a, b = int(m.group(1)), int(m.group(2))

        # Guard against out-of-range
        a = max(0, min(a, traj.n_frames - 1))
        b = max(a + 1, min(b, traj.n_frames))
        sub = traj[a:b]

        rg = md.compute_rg(sub).astype(float)  # (n_frames,)
        # Explicit reference: first frame of the slice
        ref = sub[0]
        rmsd = md.rmsd(sub, ref).astype(float)

        if rg.shape[0] != rmsd.shape[0]:
            raise ValueError("Computed CV arrays have different lengths")

        # Build a simple discrete trajectory via 2D binning (for MSM)
        b_rg = int(bins.get("rg", 32))
        b_rmsd = int(bins.get("rmsd", 32))
        # Guard against degenerate ranges
        eps = 1e-8
        rg_min, rg_max = float(np.min(rg)), float(np.max(rg))
        if abs(rg_max - rg_min) < eps:
            rg_max = rg_min + eps
        rmsd_min, rmsd_max = float(np.min(rmsd)), float(np.max(rmsd))
        if abs(rmsd_max - rmsd_min) < eps:
            rmsd_max = rmsd_min + eps
        rg_edges = np.linspace(rg_min, rg_max, b_rg + 1)
        rmsd_edges = np.linspace(rmsd_min, rmsd_max, b_rmsd + 1)
        ii = np.clip(np.digitize(rg, rg_edges) - 1, 0, b_rg - 1)
        jj = np.clip(np.digitize(rmsd, rmsd_edges) - 1, 0, b_rmsd - 1)
        dtraj = (ii * b_rmsd + jj).astype(np.int32)

        cvs = {"rg": rg, "rmsd": rmsd}
        source = {"pdb": str(PDB), "dcd": str(DCD), "range": [a, b]}
        return cvs, dtraj, source

    return _extract, {"rg", "rmsd"}


def main() -> None:
    # 0) Validate inputs exist
    if not PDB.exists() or not DCD.exists():
        raise FileNotFoundError(
            f"Missing test assets. Expected PDB={PDB} and DCD={DCD}."
        )

    # 1) Load trajectory once, plan slices, and emit shards
    import mdtraj as md

    traj = md.load(str(DCD), top=str(PDB))
    slice_paths = _make_slice_descriptors(n_frames=traj.n_frames, n_slices=3)
    applied_bins = {"rg": 32, "rmsd": 32}
    extractor, expected_cvs = _make_extractor_closure(traj, applied_bins)
    periodic_by_cv = {"rg": False, "rmsd": False}
    if set(periodic_by_cv.keys()) != expected_cvs:
        raise ValueError(
            f"periodic_by_cv keys {set(periodic_by_cv.keys())} do not match expected CVs {expected_cvs}"
        )
    jsons = emit_shards_from_trajectories(
        traj_files=slice_paths,
        out_dir=SHARDS_DIR,
        extract_cvs=extractor,
        seed_start=123,
        temperature=300.0,
        periodic_by_cv=periodic_by_cv,
    )

    # 2) Aggregate and build a provenance-first envelope
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 0.6}),))
    opts = BuildOpts(seed=123, temperature=300.0, lag_candidates=[2, 4, 6])
    applied = AppliedOpts(bins=applied_bins, lag=2, notes={"fes": {"smooth": True}})

    result, ds_hash = aggregate_and_build(
        jsons, opts=opts, plan=plan, applied=applied, out_bundle=BUNDLE
    )

    print("dataset_hash:", ds_hash)
    print("digest:", result.metadata.digest)
    print("flags:", getattr(result, "flags", {}))
    print("bundle:", BUNDLE)

    # Basic sanity checks
    if result.stationary_distribution is not None:
        s = float(np.sum(result.stationary_distribution))
        print("pi_sum:", s)
    if getattr(result, "fes", None) is None:
        print("warning: FES not present")


if __name__ == "__main__":
    main()
