import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from deeptime.markov.tools.analysis import expected_counts_stationary

from pmarlo.utils.path_utils import ensure_directory


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
            ensure_directory(out_dir)
            p = out_dir / "simulation" / "traj.dcd"
            ensure_directory(p.parent)
            p.write_bytes(b"")
            return str(p)

        def extract_features(self, _traj):
            return dummy_states

    with patch("pmarlo.experiments.simulation.Pipeline") as MockPipe:
        pipe = MockPipe.return_value
        pipe.setup_protein.return_value = MagicMock()
        pipe.setup_simulation.return_value = DummySim()
        pipe.prepared_pdb = Path("tests/_assets/3gd8-fixed.pdb")

        cfg = SimulationConfig(
            pdb_file="tests/_assets/3gd8-fixed.pdb",
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
    """Test replica exchange experiment benchmark output structure."""
    pytest.importorskip("openmm")

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
            pdb_file="tests/_assets/3gd8-fixed.pdb",
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
    """Test MSM experiment benchmark output structure."""
    pytest.importorskip("mdtraj")

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
        ensure_directory(out_dir / "msm")
        return DummyMSMObj()

    with patch("pmarlo.experiments.msm.run_complete_msm_analysis", dummy_run_complete):
        cfg = MSMConfig(
            trajectory_files=["tests/_assets/traj.dcd"],
            topology_file="tests/_assets/3gd8-fixed.pdb",
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


def test_compute_detailed_balance_mad_uses_deeptime_flows():
    from pmarlo.experiments.kpi import compute_detailed_balance_mad

    transition_matrix = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=float)
    stationary = np.array([3.0, 2.0], dtype=float)

    score = compute_detailed_balance_mad(transition_matrix, stationary)

    assert score is not None

    normalized = stationary / stationary.sum()
    flows = expected_counts_stationary(transition_matrix, 1, mu=normalized)
    manual = np.mean(np.abs(flows - flows.T)) / np.sum(flows)

    assert score == pytest.approx(manual)


def test_compute_spectral_gap():
    """Test spectral gap computation with a simple 2-state system."""
    from pmarlo.experiments.kpi import compute_spectral_gap

    # Simple 2-state system with known eigenvalues
    mat = np.array([[0.9, 0.1], [0.2, 0.8]], dtype=float)
    gap = compute_spectral_gap(mat)

    # For this system, eigenvalues are 1.0 and 0.7
    # Spectral gap = 1.0 - 0.7 = 0.3
    assert 0.25 < gap < 0.35  # Allow some numerical tolerance


def test_compute_stationary_entropy():
    """Test stationary entropy computation."""
    from pmarlo.experiments.kpi import compute_stationary_entropy

    # Uniform distribution has maximum entropy
    pi_uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    entropy_uniform = compute_stationary_entropy(pi_uniform)

    # Entropy of uniform distribution over 4 states = log2(4) ≈ 1.386 nats
    assert 1.35 < entropy_uniform < 1.40

    # Delta distribution (all mass on one state) has zero entropy
    pi_delta = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    entropy_delta = compute_stationary_entropy(pi_delta)
    assert entropy_delta == pytest.approx(0.0, abs=1e-10)
