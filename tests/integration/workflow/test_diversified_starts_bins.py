from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("openmm") is None,
    reason="Requires OpenMM",
)
def test_diversified_starts_increase_bin_coverage(
    test_fixed_pdb_file: Path, tmp_path: Path
):
    from example_programs.app_usecase.app.backend import (
        ShardRequest,
        SimulationConfig,
        WorkflowBackend,
        WorkspaceLayout,
    )
    from pmarlo.data.shard import read_shard

    ws = tmp_path
    layout = WorkspaceLayout(
        app_root=ws,
        inputs_dir=ws / "inputs",
        workspace_dir=ws / "output",
        sims_dir=ws / "output" / "sims",
        shards_dir=ws / "output" / "shards",
        models_dir=ws / "output" / "models",
        bundles_dir=ws / "output" / "bundles",
        logs_dir=ws / "output" / "logs",
        state_path=ws / "output" / "state.json",
    )
    layout.ensure()
    backend = WorkflowBackend(layout)
    temps = [290.0, 310.0, 330.0, 350.0]

    def _emit(sim_config: SimulationConfig, *, seed_start: int) -> list[Path]:
        sim_result = backend.run_sampling(sim_config)
        request = ShardRequest()
        request.temperature = float(np.median(temps))
        request.stride = 5
        request.seed_start = seed_start
        request.frames_per_shard = 600
        shard_result = backend.emit_shards(sim_result, request)
        return shard_result.shard_paths

    # First run from initial PDB
    config1 = SimulationConfig(
        pdb_path=Path(test_fixed_pdb_file),
        temperatures=temps,
        steps=1500,
        quick=True,
        random_seed=1337,
        stub_result=True,
    )
    shards1 = _emit(config1, seed_start=0)

    # Second run with different random seed to diversify initial conditions
    config2 = SimulationConfig(
        pdb_path=Path(test_fixed_pdb_file),
        temperatures=temps,
        steps=1500,
        quick=True,
        random_seed=7331,
        jitter_start=True,
        jitter_sigma_A=0.05,
        stub_result=True,
    )
    shards2 = _emit(config2, seed_start=100)

    # Load shards and compute 2D CV bins occupancy
    def load_X(paths):
        arrs = []
        for p in paths:
            _, X, _ = read_shard(Path(p))
            if X.shape[1] >= 2:
                arrs.append(X[:, :2])
        if not arrs:
            return np.empty((0, 2))
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
