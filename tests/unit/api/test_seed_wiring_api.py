from __future__ import annotations

from pathlib import Path
from typing import Any


def test_run_replica_exchange_propagates_seed(monkeypatch, tmp_path: Path):
    """Ensure api.run_replica_exchange passes seed into RemdConfig."""
    from pmarlo import api as _api

    captured: dict[str, Any] = {}

    class _FakeRemd:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.trajectory_files = [str(tmp_path / "fake.dcd")]

        @classmethod
        def from_config(cls, cfg):  # type: ignore[no-untyped-def]
            captured["random_seed"] = getattr(cfg, "random_seed", None)
            return cls()

        def plan_reporter_stride(self, *a: Any, **k: Any) -> int:
            return 1

        def setup_replicas(self, *a: Any, **k: Any) -> None:
            return None

        def run_simulation(self, *a: Any, **k: Any) -> None:
            return None

        def demux_trajectories(self, *a: Any, **k: Any):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(_api, "ReplicaExchange", _FakeRemd)

    out = _api.run_replica_exchange(
        pdb_file=str(tmp_path / "model.pdb"),
        output_dir=str(tmp_path / "out"),
        temperatures=[300.0, 310.0],
        total_steps=100,
        random_seed=4242,
    )

    assert captured.get("random_seed") == 4242
    assert isinstance(out, tuple) and isinstance(out[0], list)
