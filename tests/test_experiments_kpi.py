import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _assert_benchmark_schema(obj: dict):
    # Ensure top-level keys exist in order
    keys = list(obj.keys())
    assert keys == [
        "algorithm",
        "experiment_id",
        "input_parameters",
        "kpi_metrics",
        "notes",
        "errors",
    ]

    # Validate kpi_metrics keys and types
    kpi = obj["kpi_metrics"]
    assert list(kpi.keys()) == [
        "conformational_coverage",
        "transition_matrix_accuracy",
        "replica_exchange_success_rate",
        "runtime_seconds",
        "memory_mb",
    ]
    # Values can be float or None
    for v in kpi.values():
        assert v is None or isinstance(v, (int, float))


def test_simulation_experiment_benchmark(tmp_path: Path):
    # Arrange
    out_dir = tmp_path / "experiments_output" / "simulation"

    # Mock pipeline internals to avoid heavy deps
    from pmarlo.experiments.simulation import (
        SimulationConfig,
        run_simulation_experiment,
    )

    # Prepare dummy trajectory and states
    dummy_states = np.array([0, 1, 1, 2, 2, 2])

    class DummySim:
        def __init__(self):
            self.output_dir = out_dir

        def prepare_system(self):
            return object(), None

        def run_production(self, *_args, **_kwargs):
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / "simulation" / "traj.dcd"
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
            output_dir=str(out_dir),
            steps=10,
            use_metadynamics=False,
        )

        # Act
        result = run_simulation_experiment(cfg)

    # Assert
    run_dir = Path(result["run_dir"])
    bench = _read_json(run_dir / "benchmark.json")
    _assert_benchmark_schema(bench)
    assert bench["algorithm"] == "simulation"
    assert bench["experiment_id"] == run_dir.name
    assert bench["kpi_metrics"]["conformational_coverage"] is not None
    # With quick MSM, transition_matrix_accuracy should be present
    assert bench["kpi_metrics"]["transition_matrix_accuracy"] is not None


def test_replica_exchange_experiment_benchmark(tmp_path: Path):
    out_dir = tmp_path / "experiments_output" / "replica_exchange"

    from pmarlo.experiments.replica_exchange import (
        ReplicaExchangeConfig,
        run_replica_exchange_experiment,
    )

    class DummyREMD:
        def __init__(self, *args, **kwargs):
            pass

        def setup_replicas(self, **_):
            pass

        def run_simulation(self, **_):
            pass

        def get_exchange_statistics(self):
            return {
                "total_exchange_attempts": 10,
                "total_exchanges_accepted": 4,
                "overall_acceptance_rate": 0.4,
            }

    with patch("pmarlo.experiments.replica_exchange.ReplicaExchange", DummyREMD):
        cfg = ReplicaExchangeConfig(
            pdb_file="tests/data/3gd8-fixed.pdb",
            output_dir=str(out_dir),
            total_steps=10,
            equilibration_steps=2,
            exchange_frequency=5,
            use_metadynamics=False,
        )

        result = run_replica_exchange_experiment(cfg)

    run_dir = Path(result["run_dir"])
    bench = _read_json(run_dir / "benchmark.json")
    _assert_benchmark_schema(bench)
    assert bench["algorithm"] == "replica_exchange"
    assert bench["kpi_metrics"]["replica_exchange_success_rate"] == 0.4


def test_msm_experiment_benchmark(tmp_path: Path):
    out_dir = tmp_path / "experiments_output" / "msm"

    from pmarlo.experiments.msm import MSMConfig, run_msm_experiment

    class DummyMSMObj:
        def __init__(self):
            self.n_states = 5
            self.transition_matrix = np.array(
                [
                    [0.9, 0.1, 0.0, 0.0, 0.0],
                    [0.2, 0.8, 0.0, 0.0, 0.0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            )
            self.dtrajs = [np.array([0, 1, 1, 2, 3, 4])]

    def dummy_run_complete(*_args, **_kwargs):
        # Create output directory tree
        (out_dir / "msm").mkdir(parents=True, exist_ok=True)
        return DummyMSMObj()

    with patch("pmarlo.experiments.msm.run_complete_msm_analysis", dummy_run_complete):
        cfg = MSMConfig(
            trajectory_files=["tests/data/traj.dcd"],
            topology_file="tests/data/3gd8-fixed.pdb",
            output_dir=str(out_dir),
            n_clusters=5,
            lag_time=10,
        )
        result = run_msm_experiment(cfg)

    run_dir = Path(result["run_dir"])
    bench = _read_json(run_dir / "benchmark.json")
    _assert_benchmark_schema(bench)
    assert bench["algorithm"] == "msm"
    assert bench["kpi_metrics"]["transition_matrix_accuracy"] is not None
