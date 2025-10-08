from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pmarlo.analysis import compute_analysis_debug, export_analysis_debug
from pmarlo.transform.build import BuildResult


def _fake_dataset() -> dict[str, object]:
    return {
        "__shards__": [
            {
                "id": "s0",
                "legacy_id": "s0",
                "start": 0,
                "stop": 5,
                "temperature": 300.0,
            },
            {
                "id": "s1",
                "legacy_id": "s1",
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
    assert summary["dataset_hash"] == "abc123"
    assert summary["arrays"]["transition_counts"] == "transition_counts.npy"

    counts_file = tmp_path / summary["arrays"]["transition_counts"]
    assert counts_file.exists()
    counts = np.load(counts_file)
    assert counts.shape == (3, 3)

    state_counts_file = tmp_path / summary["arrays"]["state_counts"]
    assert np.array_equal(np.load(state_counts_file), debug_data.state_counts)

