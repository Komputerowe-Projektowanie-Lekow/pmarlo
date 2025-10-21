from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pmarlo.analysis import (
    AnalysisDebugData,
    compute_analysis_debug,
    export_analysis_debug,
)
from pmarlo.transform.build import BuildResult


def _fake_dataset() -> dict[str, object]:
    return {
        "__shards__": [
            {
                "id": "s0",
                "start": 0,
                "stop": 5,
                "temperature": 300.0,
            },
            {
                "id": "s1",
                "start": 5,
                "stop": 10,
                "temperature": 300.0,
            },
        ],
        "dtrajs": [
            np.array([0, 1, 0, 1, 0], dtype=int),
            np.array([1, 2, 1, 2, 1], dtype=int),
        ],
    }


def test_compute_analysis_debug_basic_metrics() -> None:
    dataset = _fake_dataset()
    debug_data = compute_analysis_debug(dataset, lag=1)

    assert debug_data.counts.shape == (3, 3)
    assert debug_data.state_counts.tolist() == [3, 5, 2]
    summary = debug_data.summary
    assert summary["total_pairs"] == 8
    assert summary["zero_rows"] == 0
    # Warning triggered due to small pair count threshold
    assert any(w["code"] == "TOTAL_PAIRS_LT_5000" for w in summary["warnings"])
    assert summary["largest_scc_size"] == 3
    assert pytest.approx(summary["largest_scc_frame_fraction"], rel=1e-6) == 1.0
    assert summary["effective_stride_max"] == 1
    assert summary["effective_tau_frames"] == summary["tau_frames"]


def test_export_analysis_debug_writes_expected_files(tmp_path: Path) -> None:
    dataset = _fake_dataset()
    debug_data = compute_analysis_debug(dataset, lag=1)
    br = BuildResult(
        transition_matrix=np.eye(3),
        stationary_distribution=np.array([0.4, 0.4, 0.2]),
        cluster_populations=np.array([0.5, 0.3, 0.2]),
    )
    br.msm = SimpleNamespace(
        assignments={"train": np.array([0, 1, 0, 1, 0], dtype=int)},
        assignment_masks={"train": np.ones(5, dtype=bool)},
    )
    br.artifacts = {
        "feature_stats": {
            "train": {
                "feature_names": ["cv1"],
                "n_rows": 5,
                "n_features": 1,
                "finite_rows": 5,
                "non_finite_entries": 0,
                "means": [0.0],
                "stds": [1.0],
                "mins": [-1.0],
                "maxs": [1.0],
            }
        }
    }

    info = export_analysis_debug(
        output_dir=tmp_path,
        build_result=br,
        debug_data=debug_data,
        config={"lag": 1},
        dataset_hash="abc123",
    )

    summary_path: Path = info["summary"]
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["state_assignment_splits"] == ["train"]
    assert summary["stride"] == 1
    assert summary["dataset_hash"] == "abc123"
    assert summary["arrays"]["transition_counts"] == "transition_counts.npy"
    assert summary["counted_pairs"] == summary["total_pairs"]
    assert summary["expected_pairs"] == summary["total_pairs_predicted"]

    counts_file = tmp_path / summary["arrays"]["transition_counts"]
    assert counts_file.exists()
    counts = np.load(counts_file)
    assert counts.shape == (3, 3)

    state_counts_file = tmp_path / summary["arrays"]["state_counts"]
    assert np.array_equal(np.load(state_counts_file), debug_data.state_counts)

    stats_file = summary_path.parent / summary["feature_stats_file"]
    assert stats_file.exists()
    stats_payload = json.loads(stats_file.read_text(encoding="utf-8"))
    assert "train" in stats_payload

    state_ids_path = summary_path.parent / summary["arrays"]["state_ids[train]"]
    mask_path = summary_path.parent / summary["arrays"]["valid_mask[train]"]
    assert np.array_equal(np.load(state_ids_path), br.msm.assignments["train"])
    assert np.array_equal(np.load(mask_path), br.msm.assignment_masks["train"])


def test_export_analysis_debug_rejects_zero_counts(tmp_path: Path) -> None:
    debug_data = AnalysisDebugData(
        summary={"warnings": []},
        counts=np.zeros((1, 1), dtype=float),
        state_counts=np.zeros((1,), dtype=float),
        component_labels=np.zeros((1,), dtype=int),
    )
    br = BuildResult(
        transition_matrix=np.eye(1),
        stationary_distribution=np.array([1.0]),
    )
    br.fes = {"result": {"note": "dummy"}}

    with pytest.raises(
        ValueError, match="transition counts contain no observed transitions"
    ):
        export_analysis_debug(
            output_dir=tmp_path,
            build_result=br,
            debug_data=debug_data,
            config=None,
            dataset_hash="zero",
        )


def test_compute_analysis_debug_requires_shard_ids() -> None:
    dataset = {
        "__shards__": [
            {
                "start": 0,
                "stop": 5,
            }
        ],
        "dtrajs": [np.array([0, 1, 0], dtype=int)],
    }

    with pytest.raises(ValueError, match="missing required 'id'"):
        compute_analysis_debug(dataset, lag=1)
