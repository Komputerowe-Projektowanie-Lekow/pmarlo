from __future__ import annotations

from pathlib import Path

import pytest

from pmarlo_webapp.app.backend import (
    SimulationConfig,
    WorkflowBackend,
    WorkspaceLayout,
)


@pytest.mark.unit
def test_quick_mode_runs_full_simulation_by_default(
    monkeypatch,
    tmp_path: Path,
    test_fixed_pdb_file: Path,
) -> None:
    layout = WorkspaceLayout(
        app_root=tmp_path,
        inputs_dir=tmp_path / "inputs",
        workspace_dir=tmp_path / "output",
        sims_dir=tmp_path / "output" / "sims",
        shards_dir=tmp_path / "output" / "shards",
        models_dir=tmp_path / "output" / "models",
        bundles_dir=tmp_path / "output" / "bundles",
        logs_dir=tmp_path / "output" / "logs",
        state_path=tmp_path / "output" / "state.json",
    )
    layout.ensure()
    backend = WorkflowBackend(layout)

    calls: dict[str, object] = {}

    def fake_run_replica_exchange(
        pdb_file: str,
        output_dir: str,
        temperatures: list[float],
        total_steps: int,
        **kwargs: object,
    ) -> tuple[list[str], list[float]]:
        calls["called"] = True
        calls["quick"] = kwargs.get("quick")
        calls["save_final_pdb"] = kwargs.get("save_final_pdb")
        final_pdb = kwargs.get("final_pdb_path")
        rep_dir = Path(output_dir) / "replica_exchange"
        rep_dir.mkdir(parents=True, exist_ok=True)
        traj_paths: list[str] = []
        for idx, _ in enumerate(temperatures):
            traj_path = rep_dir / f"traj_{idx:02d}.dcd"
            traj_path.write_bytes(b"")
            traj_paths.append(str(traj_path))
        if final_pdb:
            final_path = Path(str(final_pdb))
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text("END\n", encoding="utf-8")
            calls["final_pdb_path"] = final_path
        return traj_paths, [float(t) for t in temperatures]

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.run_replica_exchange",
        fake_run_replica_exchange,
    )

    config = SimulationConfig(
        pdb_path=Path(test_fixed_pdb_file),
        temperatures=[300.0, 320.0, 340.0],
        steps=500,
        quick=True,
        random_seed=123,
    )

    result = backend.run_sampling(config)

    assert calls.get("called") is True
    assert calls.get("quick") is True
    assert calls.get("save_final_pdb") is False
    assert result.restart_pdb_path is None
    assert result.traj_files
    assert not backend.state.runs[-1].get("stub_result")


@pytest.mark.unit
def test_explicit_stub_flag_skips_engine(
    monkeypatch,
    tmp_path: Path,
    test_fixed_pdb_file: Path,
) -> None:
    layout = WorkspaceLayout(
        app_root=tmp_path,
        inputs_dir=tmp_path / "inputs",
        workspace_dir=tmp_path / "output",
        sims_dir=tmp_path / "output" / "sims",
        shards_dir=tmp_path / "output" / "shards",
        models_dir=tmp_path / "output" / "models",
        bundles_dir=tmp_path / "output" / "bundles",
        logs_dir=tmp_path / "output" / "logs",
        state_path=tmp_path / "output" / "state.json",
    )
    layout.ensure()
    backend = WorkflowBackend(layout)

    def fake_run_replica_exchange(
        *_args: object, **_kwargs: object
    ) -> tuple[list[str], list[float]]:
        raise AssertionError("Engine should not be called when stub_result=True")

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.run_replica_exchange",
        fake_run_replica_exchange,
    )

    config = SimulationConfig(
        pdb_path=Path(test_fixed_pdb_file),
        temperatures=[300.0, 320.0, 340.0],
        steps=200,
        quick=True,
        stub_result=True,
    )

    result = backend.run_sampling(config)

    assert result.traj_files
    assert backend.state.runs[-1].get("stub_result") is True
    assert result.restart_pdb_path is None
    assert result.restart_inputs_entry is None


@pytest.mark.unit
def test_save_restart_snapshot_creates_inputs_copy(
    monkeypatch,
    tmp_path: Path,
    test_fixed_pdb_file: Path,
) -> None:
    layout = WorkspaceLayout(
        app_root=tmp_path,
        inputs_dir=tmp_path / "inputs",
        workspace_dir=tmp_path / "output",
        sims_dir=tmp_path / "output" / "sims",
        shards_dir=tmp_path / "output" / "shards",
        models_dir=tmp_path / "output" / "models",
        bundles_dir=tmp_path / "output" / "bundles",
        logs_dir=tmp_path / "output" / "logs",
        state_path=tmp_path / "output" / "state.json",
    )
    layout.ensure()
    backend = WorkflowBackend(layout)

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend._timestamp",
        lambda: "20250101-120000",
    )

    calls: dict[str, object] = {}

    def fake_run_replica_exchange(
        pdb_file: str,
        output_dir: str,
        temperatures: list[float],
        total_steps: int,
        **kwargs: object,
    ) -> tuple[list[str], list[float]]:
        rep_dir = Path(output_dir) / "replica_exchange"
        rep_dir.mkdir(parents=True, exist_ok=True)
        traj_paths: list[str] = []
        for idx, _ in enumerate(temperatures):
            traj_path = rep_dir / f"traj_{idx:02d}.dcd"
            traj_path.write_bytes(b"")
            traj_paths.append(str(traj_path))
        final_pdb = kwargs.get("final_pdb_path")
        if final_pdb is None:
            raise AssertionError("Expected final_pdb_path when save_restart_pdb=True")
        final_path = Path(str(final_pdb))
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text("END\n", encoding="utf-8")
        calls["save_final_pdb"] = kwargs.get("save_final_pdb")
        calls["final_pdb_path"] = final_path
        return traj_paths, [float(t) for t in temperatures]

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend.run_replica_exchange",
        fake_run_replica_exchange,
    )

    config = SimulationConfig(
        pdb_path=Path(test_fixed_pdb_file),
        temperatures=[300.0, 320.0, 340.0],
        steps=500,
        quick=True,
        random_seed=111,
        save_restart_pdb=True,
    )

    result = backend.run_sampling(config)

    assert calls.get("save_final_pdb") is True
    assert result.restart_pdb_path is not None
    assert result.restart_inputs_entry is not None
    assert result.restart_pdb_path.exists()
    assert result.restart_inputs_entry.exists()
    assert calls["final_pdb_path"] == result.restart_pdb_path
    assert "3gd8-fixed" in result.restart_inputs_entry.name
    assert result.restart_inputs_entry.parent == layout.inputs_dir.resolve()
    assert backend.state.runs[-1]["restart_pdb"] == str(result.restart_pdb_path)
    assert backend.state.runs[-1]["restart_input_entry"] == str(
        result.restart_inputs_entry
    )


@pytest.mark.unit
def test_stub_restart_snapshot_written(
    monkeypatch, tmp_path: Path, test_fixed_pdb_file: Path
) -> None:
    layout = WorkspaceLayout(
        app_root=tmp_path,
        inputs_dir=tmp_path / "inputs",
        workspace_dir=tmp_path / "output",
        sims_dir=tmp_path / "output" / "sims",
        shards_dir=tmp_path / "output" / "shards",
        models_dir=tmp_path / "output" / "models",
        bundles_dir=tmp_path / "output" / "bundles",
        logs_dir=tmp_path / "output" / "logs",
        state_path=tmp_path / "output" / "state.json",
    )
    layout.ensure()
    backend = WorkflowBackend(layout)

    monkeypatch.setattr(
        "pmarlo_webapp.app.backend._timestamp",
        lambda: "20250101-120000",
    )

    config = SimulationConfig(
        pdb_path=Path(test_fixed_pdb_file),
        temperatures=[300.0, 320.0, 340.0],
        steps=200,
        quick=True,
        stub_result=True,
        save_restart_pdb=True,
    )

    result = backend.run_sampling(config)

    assert result.restart_pdb_path is not None
    assert result.restart_inputs_entry is not None
    assert result.restart_pdb_path.exists()
    assert result.restart_inputs_entry.exists()
    assert result.restart_inputs_entry.parent == layout.inputs_dir.resolve()
    assert backend.state.runs[-1]["restart_pdb"] == str(result.restart_pdb_path)
    assert backend.state.runs[-1]["restart_input_entry"] == str(
        result.restart_inputs_entry
    )
