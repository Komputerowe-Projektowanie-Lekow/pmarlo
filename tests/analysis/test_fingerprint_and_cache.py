from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from example_programs.app_usecase.app.backend import BuildArtifact, BuildConfig, WorkspaceLayout, WorkflowBackend
from pmarlo.data.shard import write_shard


def _make_workspace(tmp_path: Path) -> tuple[WorkflowBackend, list[Path]]:
    workspace_dir = tmp_path / "workspace"
    layout = WorkspaceLayout(
        app_root=tmp_path,
        inputs_dir=tmp_path / "inputs",
        workspace_dir=workspace_dir,
        sims_dir=workspace_dir / "sims",
        shards_dir=workspace_dir / "shards",
        models_dir=workspace_dir / "models",
        bundles_dir=workspace_dir / "bundles",
        logs_dir=workspace_dir / "logs",
        state_path=workspace_dir / "state.json",
    )
    layout.ensure()

    rng = np.random.default_rng(1234)
    frames = 20000
    cv1 = np.cumsum(rng.normal(size=frames))
    cv2 = np.cumsum(rng.normal(size=frames))

    states = np.zeros(frames, dtype=np.int32)
    for t in range(1, frames):
        if rng.random() < 0.7:
            states[t] = states[t - 1]
        else:
            states[t] = rng.integers(0, 10)

    shard_path = write_shard(
        out_dir=layout.shards_dir,
        shard_id="synthetic",
        cvs={"cv1": cv1, "cv2": cv2},
        dtraj=states,
        periodic={"cv1": False, "cv2": False},
        seed=2025,
        temperature=300.0,
        source={"kind": "synthetic"},
    )

    backend = WorkflowBackend(layout)
    return backend, [shard_path]


def _run_build(
    backend: WorkflowBackend,
    shard_paths: list[Path],
    n_states: int,
    lag: int,
) -> tuple[BuildArtifact, Path, dict]:
    config = BuildConfig(
        lag=lag,
        bins={"cv1": 64, "cv2": 64},
        seed=2025,
        temperature=300.0,
        learn_cv=False,
        apply_cv_whitening=True,
        cluster_mode="kmeans",
        n_microstates=n_states,
        reweight_mode="MBAR",
        fes_method="kde",
        fes_bandwidth="scott",
        fes_min_count_per_bin=1,
    )
    artifact = backend.build_analysis([Path(p) for p in shard_paths], config)
    debug_dir = Path(artifact.debug_dir or "")
    summary_data = json.loads((debug_dir / "summary.json").read_text(encoding="utf-8"))
    summary = summary_data.get("summary", summary_data)
    return artifact, debug_dir, summary


def test_fingerprint_changes_invalidate_cache(tmp_path):
    backend, shard_paths = _make_workspace(tmp_path)

    artifact_a, dir_a, summary_a = _run_build(backend, shard_paths, n_states=40, lag=3000)
    time.sleep(1.1)
    artifact_b, dir_b, summary_b = _run_build(backend, shard_paths, n_states=60, lag=3000)

    assert summary_a["fingerprint"]["n_states"] == 40
    assert summary_b["fingerprint"]["n_states"] == 60

    tm_a = artifact_a.build_result.transition_matrix
    tm_b = artifact_b.build_result.transition_matrix
    assert tm_a is not None and tm_b is not None
    assert tm_a.shape[0] != tm_b.shape[0]
    assert not np.array_equal(tm_a, tm_b)

    counts_a = np.load(dir_a / "transition_counts.npy")
    counts_b = np.load(dir_b / "transition_counts.npy")
    assert counts_a.shape == counts_b.shape
