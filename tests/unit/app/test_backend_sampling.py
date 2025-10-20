from __future__ import annotations

from pathlib import Path

import pytest

from example_programs.app_usecase.app.backend import (
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
        rep_dir = Path(output_dir) / "replica_exchange"
        rep_dir.mkdir(parents=True, exist_ok=True)
        traj_paths: list[str] = []
        for idx, _ in enumerate(temperatures):
            traj_path = rep_dir / f"traj_{idx:02d}.dcd"
            traj_path.write_bytes(b"")
            traj_paths.append(str(traj_path))
        return traj_paths, [float(t) for t in temperatures]

    monkeypatch.setattr(
        "example_programs.app_usecase.app.backend.run_replica_exchange",
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

    def fake_run_replica_exchange(*_args: object, **_kwargs: object) -> tuple[list[str], list[float]]:
        raise AssertionError("Engine should not be called when stub_result=True")

    monkeypatch.setattr(
        "example_programs.app_usecase.app.backend.run_replica_exchange",
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
