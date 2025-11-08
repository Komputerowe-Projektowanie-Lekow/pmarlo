from __future__ import annotations

from pathlib import Path
from typing import Any


def test_run_single_temperature_md_propagates_seed(monkeypatch, tmp_path: Path):
    """Ensure api.run_single_temperature_md passes seed into RemdConfig."""
    from pmarlo.api import single_temp_md as _stmd

    captured: dict[str, Any] = {}

    class _FakeRemd:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.trajectory_files = [str(tmp_path / "fake.dcd")]
            self.n_replicas = 1

        @classmethod
        def from_config(cls, cfg):  # type: ignore[no-untyped-def]
            captured["random_seed"] = getattr(cfg, "random_seed", None)
            captured["temperatures"] = getattr(cfg, "temperatures", None)
            captured["exchange_frequency"] = getattr(cfg, "exchange_frequency", None)
            return cls()

        def plan_reporter_stride(self, *a: Any, **k: Any) -> int:
            return 1

        def setup_replicas(self, *a: Any, **k: Any) -> None:
            return None

        def run_simulation(self, *a: Any, **k: Any) -> None:
            captured["simulation_called"] = True
            return None

        def export_current_structure(self, destination: str | Path, **_k: Any) -> Path:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("END\n", encoding="utf-8")
            return path

    monkeypatch.setattr(_stmd, "ReplicaExchange", _FakeRemd)

    traj_files, temp = _stmd.run_single_temperature_md(
        pdb_file=str(tmp_path / "model.pdb"),
        output_dir=str(tmp_path / "out"),
        temperature=300.0,
        total_steps=100,
        random_seed=4242,
    )

    assert captured.get("random_seed") == 4242
    assert captured.get("temperatures") == [300.0], "Should have single temperature"
    assert captured.get("exchange_frequency") == 99999999, "Exchange frequency should be very large"
    assert captured.get("simulation_called") is True, "run_simulation should be called"
    assert isinstance(traj_files, list)
    assert temp == 300.0


def test_run_single_temperature_md_exports_restart_snapshot(monkeypatch, tmp_path: Path):
    """Ensure single-temp MD can export restart snapshot."""
    from pmarlo.api import single_temp_md as _stmd

    captured: dict[str, Any] = {}

    class _FakeRemd:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.trajectory_files = [str(tmp_path / "fake.dcd")]
            self.n_replicas = 1

        @classmethod
        def from_config(cls, cfg):  # type: ignore[no-untyped-def]
            return cls()

        def plan_reporter_stride(self, *a: Any, **k: Any) -> int:
            return 1

        def setup_replicas(self, *a: Any, **k: Any) -> None:
            return None

        def run_simulation(self, *a: Any, **k: Any) -> None:
            return None

        def export_current_structure(
            self,
            destination: str | Path,
            *,
            temperature: float | None = None,
            **_k: Any,
        ) -> Path:
            captured["temperature"] = temperature
            dest = Path(destination)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("END\n", encoding="utf-8")
            captured["destination"] = dest
            return dest

    monkeypatch.setattr(_stmd, "ReplicaExchange", _FakeRemd)

    final_path = tmp_path / "out" / "snapshot.pdb"
    traj_files, temp = _stmd.run_single_temperature_md(
        pdb_file=str(tmp_path / "model.pdb"),
        output_dir=str(tmp_path / "out"),
        temperature=300.0,
        total_steps=100,
        save_final_pdb=True,
        final_pdb_path=final_path,
        final_pdb_temperature=305.0,
    )

    assert isinstance(traj_files, list)
    assert temp == 300.0
    assert captured.get("destination") == final_path
    assert captured.get("temperature") == 305.0


def test_single_temp_md_uses_single_replica(monkeypatch, tmp_path: Path):
    """Ensure single-temp MD creates exactly 1 replica."""
    from pmarlo.api import single_temp_md as _stmd

    captured: dict[str, Any] = {}

    class _FakeRemd:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.trajectory_files = [str(tmp_path / "fake.dcd")]
            self.n_replicas = 1

        @classmethod
        def from_config(cls, cfg):  # type: ignore[no-untyped-def]
            temps = getattr(cfg, "temperatures", [])
            captured["num_temperatures"] = len(temps)
            return cls()

        def plan_reporter_stride(self, *a: Any, **k: Any) -> int:
            return 1

        def setup_replicas(self, *a: Any, **k: Any) -> None:
            return None

        def run_simulation(self, *a: Any, **k: Any) -> None:
            return None

        def export_current_structure(self, destination: str | Path, **_k: Any) -> Path:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("END\n", encoding="utf-8")
            return path

    monkeypatch.setattr(_stmd, "ReplicaExchange", _FakeRemd)

    traj_files, temp = _stmd.run_single_temperature_md(
        pdb_file=str(tmp_path / "model.pdb"),
        output_dir=str(tmp_path / "out"),
        temperature=310.5,
        total_steps=100,
    )

    assert captured.get("num_temperatures") == 1, "Should create exactly 1 replica"
    assert isinstance(traj_files, list)
    assert temp == 310.5

