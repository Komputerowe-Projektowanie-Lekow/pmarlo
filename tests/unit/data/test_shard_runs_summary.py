from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmarlo.data.shard import write_shard
from pmarlo.data.shard_io import ShardRunSummary, summarize_shard_runs


def _emit_demo_shard(
    tmp_path: Path,
    *,
    run_id: str,
    segment_id: int,
    temperature: float,
) -> Path:
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / f"{run_id}_seg{segment_id:04d}.dcd"
    traj_path.write_bytes(b"")  # minimal placeholder trajectory

    cvs = {
        "phi": np.linspace(-np.pi, np.pi, 6),
        "psi": np.linspace(-np.pi, np.pi, 6),
    }
    periodic = {"phi": True, "psi": True}
    source = {
        "traj": str(traj_path),
        "run_id": run_id,
        "created_at": "1970-01-01T00:00:00Z",
        "kind": "demux",
        "segment_id": segment_id,
        "replica_id": 0,
        "exchange_window_id": 0,
    }
    shard_id = f"T{int(round(temperature))}K_seg{segment_id:04d}_rep000"
    return write_shard(
        out_dir=tmp_path,
        shard_id=shard_id,
        cvs=cvs,
        dtraj=None,
        periodic=periodic,
        seed=segment_id,
        temperature=temperature,
        source=source,
    )


def test_summarize_shard_runs_single_run(tmp_path: Path) -> None:
    shard_a = _emit_demo_shard(
        tmp_path, run_id="run-001", segment_id=1, temperature=300.0
    )
    shard_b = _emit_demo_shard(
        tmp_path, run_id="run-001", segment_id=2, temperature=300.0
    )

    summaries = summarize_shard_runs([shard_a, shard_b])
    assert len(summaries) == 1
    summary = summaries[0]
    assert isinstance(summary, ShardRunSummary)
    assert summary.run_id == "run-001"
    assert summary.temperature_K == pytest.approx(300.0)
    assert summary.shard_count == 2
    assert summary.shard_paths == (shard_a.resolve(), shard_b.resolve())


def test_summarize_shard_runs_multiple_order(tmp_path: Path) -> None:
    shard_1 = _emit_demo_shard(
        tmp_path, run_id="run-B", segment_id=1, temperature=305.0
    )
    shard_2 = _emit_demo_shard(
        tmp_path, run_id="run-A", segment_id=1, temperature=295.0
    )
    shard_3 = _emit_demo_shard(
        tmp_path, run_id="run-A", segment_id=2, temperature=295.0
    )

    summaries = summarize_shard_runs([shard_1, shard_2, shard_3])
    assert [s.run_id for s in summaries] == ["run-B", "run-A"]
    assert summaries[0].temperature_K == pytest.approx(305.0)
    assert summaries[0].shard_count == 1
    assert summaries[1].temperature_K == pytest.approx(295.0)
    assert summaries[1].shard_count == 2


def test_summarize_shard_runs_inconsistent_temperature(tmp_path: Path) -> None:
    shard_hot = _emit_demo_shard(
        tmp_path, run_id="run-mix", segment_id=1, temperature=315.0
    )
    shard_cold = _emit_demo_shard(
        tmp_path, run_id="run-mix", segment_id=2, temperature=290.0
    )

    with pytest.raises(ValueError, match="Inconsistent temperatures for run run-mix"):
        summarize_shard_runs([shard_hot, shard_cold])
