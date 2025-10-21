from __future__ import annotations

import json
import time
from itertools import count
from pathlib import Path

import numpy as np

from example_programs.app_usecase.app.backend import (
    BuildArtifact,
    BuildConfig,
    WorkflowBackend,
    WorkspaceLayout,
)
from pmarlo.data.shard import write_shard

_SEGMENT_COUNTER = count()


def _canonical_shard_id(temperature_K: float, segment_id: int, replica_id: int) -> str:
    temp = int(round(temperature_K))
    return f"T{temp}K_seg{segment_id:04d}_rep{replica_id:03d}"


def _source_metadata(segment_id: int, replica_id: int) -> dict[str, object]:
    return {
        "created_at": "1970-01-01T00:00:00Z",
        "kind": "demux",
        "run_id": "workflow-test",
        "segment_id": int(segment_id),
        "replica_id": int(replica_id),
        "exchange_window_id": 0,
    }


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

    segment_id = next(_SEGMENT_COUNTER)
    replica_id = 0
    shard_id = _canonical_shard_id(300.0, segment_id, replica_id)
    shard_path = write_shard(
        out_dir=layout.shards_dir,
        shard_id=shard_id,
        cvs={"cv1": cv1, "cv2": cv2},
        dtraj=states,
        periodic={"cv1": False, "cv2": False},
        seed=2025,
        temperature=300.0,
        source=_source_metadata(segment_id, replica_id),
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

    artifact_a, dir_a, summary_a = _run_build(
        backend, shard_paths, n_states=40, lag=3000
    )
    time.sleep(1.1)
    artifact_b, dir_b, summary_b = _run_build(
        backend, shard_paths, n_states=60, lag=3000
    )

    assert summary_a["fingerprint"]["n_states"] == 40
    assert summary_b["fingerprint"]["n_states"] == 60

    assert dir_a != dir_b

    notes_a = artifact_a.build_result.metadata.applied_opts.notes
    notes_b = artifact_b.build_result.metadata.applied_opts.notes
    overrides_a = notes_a.get("analysis_overrides", {})
    overrides_b = notes_b.get("analysis_overrides", {})
    assert overrides_a.get("n_microstates") == 40
    assert overrides_b.get("n_microstates") == 60

    counts_a = np.load(dir_a / "transition_counts.npy")
    counts_b = np.load(dir_b / "transition_counts.npy")
    assert counts_a.shape == counts_b.shape
