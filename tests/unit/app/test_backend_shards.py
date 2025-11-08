from __future__ import annotations

import json
from pathlib import Path

import pytest

from pmarlo_webapp.app.backend import Backend, WorkspaceLayout


def _build_layout(tmp_path: Path) -> WorkspaceLayout:
    layout = WorkspaceLayout(
        app_root=tmp_path,
        inputs_dir=tmp_path / "inputs",
        workspace_dir=tmp_path / "workspace",
        sims_dir=tmp_path / "workspace" / "sims",
        shards_dir=tmp_path / "workspace" / "shards",
        models_dir=tmp_path / "workspace" / "models",
        bundles_dir=tmp_path / "workspace" / "bundles",
        logs_dir=tmp_path / "workspace" / "logs",
        state_path=tmp_path / "workspace" / "state.json",
    )
    layout.ensure()
    return layout


def _write_shard_json(path: Path, n_frames: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_frames": n_frames}), encoding="utf-8")
    return str(path)


@pytest.mark.unit
def test_shard_summaries_merge_duplicate_runs(tmp_path: Path) -> None:
    layout = _build_layout(tmp_path)
    backend = Backend(layout)

    shard_a = _write_shard_json(layout.shards_dir / "run-1" / "a.json", 100)
    shard_b = _write_shard_json(layout.shards_dir / "run-1" / "b.json", 50)
    shard_c = _write_shard_json(layout.shards_dir / "run-2" / "c.json", 75)

    backend.state.append_shards(
        {
            "run_id": "run-1",
            "paths": [shard_a],
            "n_frames": 100,
            "temperature": 310.0,
            "analysis_temperatures": [310.0],
            "created_at": "20240101-000000",
            "cv_informed": False,
        }
    )
    backend.state.append_shards(
        {
            "run_id": "run-1",
            "paths": [shard_b],
            "n_frames": 50,
            "temperature": 310.0,
            "analysis_temperatures": [320.0],
            "created_at": "20240202-000000",
            "cv_informed": True,
        }
    )
    backend.state.append_shards(
        {
            "run_id": "run-2",
            "paths": [shard_c],
            "n_frames": 75,
            "temperature": 300.0,
            "analysis_temperatures": [300.0],
            "created_at": "20240115-000000",
            "cv_informed": False,
        }
    )

    summaries = backend.shard_summaries()
    assert len(summaries) == 2

    summary_map = {entry["run_id"]: entry for entry in summaries}

    run_one = summary_map["run-1"]
    assert run_one["n_frames"] == 150
    assert run_one["n_shards"] == 2
    assert set(run_one["paths"]) == {shard_a, shard_b}
    assert run_one["cv_informed"] is True  # merged flag
    assert run_one["analysis_temperatures"] == [310.0, 320.0]
    assert run_one["created_at"] == "20240202-000000"

    run_two = summary_map["run-2"]
    assert run_two["n_frames"] == 75
    assert run_two["paths"] == [shard_c]


@pytest.mark.unit
def test_shard_summaries_reject_temperature_conflicts(tmp_path: Path) -> None:
    layout = _build_layout(tmp_path)
    backend = Backend(layout)

    shard_a = _write_shard_json(layout.shards_dir / "run-x" / "a.json", 10)
    shard_b = _write_shard_json(layout.shards_dir / "run-x" / "b.json", 20)

    backend.state.append_shards(
        {
            "run_id": "run-x",
            "paths": [shard_a],
            "n_frames": 10,
            "temperature": 300.0,
            "created_at": "20240101-000000",
        }
    )
    backend.state.append_shards(
        {
            "run_id": "run-x",
            "paths": [shard_b],
            "n_frames": 20,
            "temperature": 305.0,
            "created_at": "20240102-000000",
        }
    )

    with pytest.raises(ValueError, match="Inconsistent temperatures recorded for run"):
        backend.shard_summaries()
