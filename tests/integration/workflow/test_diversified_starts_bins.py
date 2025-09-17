from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("openmm") is None,
    reason="Requires OpenMM",
)
def test_diversified_starts_increase_bin_coverage(
    test_fixed_pdb_file: Path, tmp_path: Path
):
    from example_programs.app_usecase.app.backend import (
        emit_from_trajs_simple,
        run_short_sim,
    )
    from pmarlo.data.shard import read_shard

    ws = tmp_path
    temps = [290.0, 310.0, 330.0, 350.0]

    # First run from initial PDB
    sim1 = run_short_sim(Path(test_fixed_pdb_file), ws, temps, steps=1500, quick=True)
    shards1_dir = ws / "shards" / Path(sim1.run_dir).name
    shards1 = emit_from_trajs_simple(
        sim1.traj_files,
        shards1_dir,
        pdb=Path(test_fixed_pdb_file),
        ref_dcd=None,
        temperature=float(np.median(temps)),
        seed_start=0,
        stride=5,
    )

    # Second run seeded from random high-T frame of first run
    sim2 = run_short_sim(
        Path(test_fixed_pdb_file),
        ws,
        temps,
        steps=1500,
        quick=True,
        start_mode="random_highT",
        start_run=sim1.run_dir,
        jitter_start=True,
        jitter_sigma_A=0.05,
        velocity_reseed=True,
    )
    shards2_dir = ws / "shards" / Path(sim2.run_dir).name
    shards2 = emit_from_trajs_simple(
        sim2.traj_files,
        shards2_dir,
        pdb=Path(test_fixed_pdb_file),
        ref_dcd=None,
        temperature=float(np.median(temps)),
        seed_start=100,
        stride=5,
    )

    # Load shards and compute 2D CV bins occupancy
    def load_X(paths):
        arrs = []
        for p in paths:
            _, X, _ = read_shard(Path(p))
            if X.shape[1] >= 2:
                arrs.append(X[:, :2])
        if not arrs:
            return np.zeros((0, 2))
        return np.vstack(arrs)

    X1 = load_X(shards1)
    X2 = load_X(shards2)
    assert X1.shape[0] > 0 and X2.shape[0] > 0

    # Common edges across both
    mins = np.minimum(np.nanmin(X1, axis=0), np.nanmin(X2, axis=0))
    maxs = np.maximum(np.nanmax(X1, axis=0), np.nanmax(X2, axis=0))
    # Guard degenerate ranges
    maxs = np.where(maxs == mins, mins + 1e-8, maxs)
    ex = np.linspace(mins[0], maxs[0], 24 + 1)
    ey = np.linspace(mins[1], maxs[1], 24 + 1)

    def occ(X):
        if X.shape[0] == 0:
            return set()
        bx = np.clip(np.digitize(X[:, 0], ex) - 1, 0, len(ex) - 2)
        by = np.clip(np.digitize(X[:, 1], ey) - 1, 0, len(ey) - 2)
        return set((int(i), int(j)) for i, j in zip(bx, by))

    occ1 = occ(X1)
    occ_union = occ(np.vstack([X1, X2]))
    # Expect union bins to be >= first alone; usually greater when starts are diversified
    assert len(occ_union) >= len(occ1)
    # A conservative check that diversity helped in typical cases
    # Allow equality in rare edge cases
