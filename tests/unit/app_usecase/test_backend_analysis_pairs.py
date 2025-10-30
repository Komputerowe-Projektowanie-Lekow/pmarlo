from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pmarlo.transform.build import AppliedOpts, BuildResult, RunMetadata
from pmarlo_webapp.app.backend import (
    BuildConfig,
    WorkflowBackend,
    WorkspaceLayout,
)


@pytest.fixture
def workspace(tmp_path: Path) -> WorkspaceLayout:
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
    return layout


def test_analysis_total_pairs_matches_summary(
    monkeypatch: pytest.MonkeyPatch, workspace: WorkspaceLayout
) -> None:
    backend = WorkflowBackend(workspace)

    shard_paths = [workspace.shards_dir / f"shard-{idx}.json" for idx in range(3)]
    for path in shard_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    lengths = [6, 5, 7]
    total_frames = sum(lengths)
    dataset = {
        "__shards__": [
            {
                "id": f"shard-{idx}",
                "length": length,
                "frames_loaded": length,
                "frames_declared": length,
                "effective_frame_stride": 1,
            }
            for idx, length in enumerate(lengths)
        ],
        "X": np.zeros((total_frames, 2), dtype=float),
    }

    def fake_load_shards_as_dataset(paths: list[Path]):
        assert len(paths) == len(shard_paths)
        return dataset

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.load_shards_as_dataset",
        fake_load_shards_as_dataset,
    )

    counts = np.array(
        [
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums > 0,
    )
    stationarity = np.array([0.4, 0.3, 0.3], dtype=float)
    per_shard_pairs = [3, 4, 5]
    total_pairs = sum(per_shard_pairs)
    counted_pairs = {f"shard-{idx}": value for idx, value in enumerate(per_shard_pairs)}
    counted_pairs["all"] = total_pairs
    state_counts = np.array([10.0, 9.0, 8.0], dtype=float)

    msm_payload = {
        "counts": counts,
        "state_counts": state_counts,
        "counted_pairs": counted_pairs,
        "feature_schema": {"names": ["cv1", "cv2"], "n_features": 2},
        "lag_time": 1,
    }

    expected_microstates = 3

    def fake_build_from_shards(**kwargs):
        assert kwargs.get("n_microstates") == expected_microstates
        notes = kwargs.get("notes")
        applied_opts = AppliedOpts()
        if isinstance(notes, dict):
            applied_opts.notes = notes
        metadata = RunMetadata(
            run_id="test",
            start_time="1970-01-01T00:00:00Z",
            success=True,
            applied_opts=applied_opts,
            seed=int(kwargs.get("seed", 0)),
            temperature=float(kwargs.get("temperature", 0.0)),
        )
        build_result = BuildResult(
            transition_matrix=transition_matrix,
            stationary_distribution=stationarity,
            msm=msm_payload,
            metadata=metadata,
            applied_opts=applied_opts,
            n_frames=total_frames,
            n_shards=len(lengths),
            feature_names=["cv1", "cv2"],
            artifacts={},
            flags={},
            diagnostics={},
        )
        return build_result, "fake-hash"

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.build_from_shards",
        fake_build_from_shards,
    )

    config = BuildConfig(
        lag=1,
        bins={"cv1": 8, "cv2": 8},
        seed=7,
        temperature=300.0,
        cluster_mode="kmeans",
        n_microstates=expected_microstates,
        kmeans_kwargs={},
    )

    artifact = backend.build_analysis([Path(p) for p in shard_paths], config)

    assert artifact.debug_summary is not None
    assert artifact.debug_summary["total_pairs"] == total_pairs
    assert artifact.debug_summary["counted_pairs"] == total_pairs

    flags = artifact.build_result.flags
    assert flags["analysis_total_pairs"] == total_pairs
    assert artifact.build_result.metadata is not None
    assert artifact.build_result.metadata.applied_opts is not None
    assert (
        artifact.build_result.metadata.applied_opts.notes["analysis_total_pairs"]
        == total_pairs
    )

    assert artifact.analysis_healthy is True
    assert not artifact.guardrail_violations


def test_analysis_state_count_guardrail(
    monkeypatch: pytest.MonkeyPatch, workspace: WorkspaceLayout
) -> None:
    backend = WorkflowBackend(workspace)

    shard_paths = [workspace.shards_dir / f"shard-{idx}.json" for idx in range(2)]
    for path in shard_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    lengths = [12, 8]
    total_frames = sum(lengths)
    dataset = {
        "__shards__": [
            {
                "id": f"shard-{idx}",
                "length": length,
                "frames_loaded": length,
                "frames_declared": length,
                "effective_frame_stride": 1,
            }
            for idx, length in enumerate(lengths)
        ],
        "X": np.zeros((total_frames, 2), dtype=float),
    }

    def fake_load_shards_as_dataset(paths: list[Path]):
        assert len(paths) == len(shard_paths)
        return dataset

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.load_shards_as_dataset",
        fake_load_shards_as_dataset,
    )

    counts = np.array(
        [
            [0.0, 5.0, 1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 1.0],
            [1.0, 0.0, 2.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        counts,
        row_sums,
        out=np.zeros_like(counts),
        where=row_sums > 0,
    )
    actual_state_count = counts.shape[0]
    state_counts = counts.sum(axis=1)
    per_shard_pairs = [int(row_sums[idx].sum()) for idx in range(len(shard_paths))]
    total_pairs = int(counts.sum())
    counted_pairs = {f"shard-{idx}": value for idx, value in enumerate(per_shard_pairs)}
    counted_pairs["all"] = total_pairs

    msm_payload = {
        "counts": counts,
        "state_counts": state_counts,
        "counted_pairs": counted_pairs,
        "feature_schema": {"names": ["cv1", "cv2"], "n_features": 2},
        "lag_time": 1,
    }

    expected_microstates = 3

    def fake_build_from_shards(**kwargs):
        assert kwargs.get("n_microstates") == expected_microstates
        notes = kwargs.get("notes")
        applied_opts = AppliedOpts()
        if isinstance(notes, dict):
            applied_opts.notes = notes
        metadata = RunMetadata(
            run_id="mismatch-run",
            start_time="1970-01-01T00:00:00Z",
            success=True,
            applied_opts=applied_opts,
            seed=int(kwargs.get("seed", 0)),
            temperature=float(kwargs.get("temperature", 0.0)),
        )
        build_result = BuildResult(
            transition_matrix=transition_matrix,
            stationary_distribution=np.full(
                actual_state_count, 1.0 / actual_state_count
            ),
            msm=msm_payload,
            metadata=metadata,
            applied_opts=applied_opts,
            n_frames=total_frames,
            n_shards=len(lengths),
            feature_names=["cv1", "cv2"],
            artifacts={},
            flags={},
            diagnostics={},
        )
        return build_result, "fake-hash"

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.build_from_shards",
        fake_build_from_shards,
    )

    config = BuildConfig(
        lag=1,
        bins={"cv1": 8, "cv2": 8},
        seed=11,
        temperature=300.0,
        cluster_mode="kmeans",
        n_microstates=expected_microstates,
        kmeans_kwargs={},
    )

    with pytest.raises(ValueError) as excinfo:
        backend.build_analysis([Path(p) for p in shard_paths], config)

    assert "state_count_mismatch" in str(excinfo.value)

    debug_dirs = sorted(workspace.analysis_debug_dir.glob("analysis-*"))
    assert (
        debug_dirs
    ), "Expected analysis debug directory to be created on guardrail failure"
    summary_path = debug_dirs[-1] / "summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["analysis_healthy"] is False
    assert summary["analysis_msm_n_states"] == actual_state_count
    guardrails = summary.get("analysis_guardrail_violations") or []
    mismatch_entry = next(
        (entry for entry in guardrails if entry.get("code") == "state_count_mismatch"),
        None,
    )
    assert mismatch_entry is not None
    assert mismatch_entry.get("declared") == config.n_microstates
    assert mismatch_entry.get("actual") == actual_state_count
    assert f"declared={config.n_microstates}" in str(mismatch_entry.get("message", ""))
