from __future__ import annotations

from pathlib import Path

from pmarlo.replica_exchange.config import RemdConfig
from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


def test_explicit_temperature_vector_sets_replicas(test_pdb_file: Path, tmp_path: Path):
    temps = [300.0 + i for i in range(12)]
    cfg = RemdConfig(
        pdb_file=str(test_pdb_file),
        temperatures=temps,
        output_dir=str(tmp_path / "out"),
        exchange_frequency=100,
    )
    remd = ReplicaExchange.from_config(cfg)
    assert remd.n_replicas == 12
    assert [float(t) for t in remd.temperatures] == [float(t) for t in temps]


def test_vector_overrides_any_automatic_ladder(test_pdb_file: Path, tmp_path: Path):
    temps = [300.0, 400.0, 500.0, 700.0]
    cfg = RemdConfig(
        pdb_file=str(test_pdb_file),
        temperatures=temps,
        output_dir=str(tmp_path / "out2"),
        exchange_frequency=100,
    )
    remd = ReplicaExchange.from_config(cfg)
    assert remd.temperatures == temps
