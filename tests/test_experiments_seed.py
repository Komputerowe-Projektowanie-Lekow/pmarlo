from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from pmarlo.experiments.msm import MSMConfig, run_msm_experiment
from pmarlo.experiments.replica_exchange import (
    ReplicaExchangeConfig,
    run_replica_exchange_experiment,
)
from pmarlo.experiments.simulation import SimulationConfig, run_simulation_experiment


def test_simulation_experiment_uses_seed(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "pmarlo.experiments.simulation.set_seed",
        lambda s: captured.setdefault("seed", s),
    )

    dummy_states = np.array([0, 1])

    class DummySim:
        def __init__(self):
            self.output_dir = tmp_path

        def prepare_system(self):
            return object(), None

        def run_production(self, *_args, **_kwargs):
            p = tmp_path / "simulation" / "traj.dcd"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
            return str(p)

        def extract_features(self, _traj):
            return dummy_states

    with patch("pmarlo.experiments.simulation.Pipeline") as MockPipe:
        pipe = MockPipe.return_value
        pipe.setup_protein.return_value = MagicMock()
        pipe.setup_simulation.return_value = DummySim()
        pipe.prepared_pdb = Path("tests/data/3gd8-fixed.pdb")

        cfg = SimulationConfig(
            pdb_file="tests/data/3gd8-fixed.pdb",
            output_dir=str(tmp_path),
            steps=5,
            use_metadynamics=False,
            seed=123,
        )
        run_simulation_experiment(cfg)

    assert captured["seed"] == 123


def test_replica_exchange_experiment_uses_seed(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "pmarlo.experiments.replica_exchange.set_seed",
        lambda s: captured.setdefault("seed", s),
    )

    class DummyREMD:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_config(cls, cfg):
            assert cfg.random_seed == 456
            return cls()

        def setup_replicas(self, **_):
            pass

        def run_simulation(self, **_):
            pass

        def get_exchange_statistics(self):
            return {}

    with patch("pmarlo.experiments.replica_exchange.ReplicaExchange", DummyREMD):
        cfg = ReplicaExchangeConfig(
            pdb_file="tests/data/3gd8-fixed.pdb",
            output_dir=str(tmp_path),
            total_steps=10,
            equilibration_steps=2,
            exchange_frequency=5,
            use_metadynamics=False,
            seed=456,
        )
        run_replica_exchange_experiment(cfg)

    assert captured["seed"] == 456


def test_msm_experiment_uses_seed(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        "pmarlo.experiments.msm.set_seed", lambda s: captured.setdefault("seed", s)
    )

    class DummyMSMObj:
        def __init__(self):
            import numpy as np

            self.n_states = 2
            self.transition_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
            self.dtrajs = [np.array([0, 1, 0, 1])]
            self.lag_time = 1

    def dummy_run_complete(*_args, **_kwargs):
        (tmp_path / "msm").mkdir(parents=True, exist_ok=True)
        return DummyMSMObj()

    with patch("pmarlo.experiments.msm.run_complete_msm_analysis", dummy_run_complete):
        cfg = MSMConfig(
            trajectory_files=["tests/data/traj.dcd"],
            topology_file="tests/data/3gd8-fixed.pdb",
            output_dir=str(tmp_path),
            n_clusters=5,
            lag_time=10,
            seed=789,
        )
        run_msm_experiment(cfg)

    assert captured["seed"] == 789
