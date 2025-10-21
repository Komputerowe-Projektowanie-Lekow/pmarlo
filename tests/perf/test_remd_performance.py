from __future__ import annotations

"""Algorithm-focused performance benchmarks for REMD diagnostics.

These benchmarks favor lightweight, deterministic micro-benchmarks over full
simulation workflows so that we can exercise the critical algorithms quickly.

Run with: pytest -m benchmark tests/perf/test_remd_performance.py
"""

import os
from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit
from openmm.app import ForceField, PDBFile, Simulation

from pmarlo.replica_exchange.system_builder import create_system

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.replica]

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


@pytest.fixture
def benchmark_protein_inputs(test_fixed_pdb_file: Path) -> tuple[PDBFile, ForceField]:
    """Provide the canonical benchmark topology for OpenMM-based perf tests."""

    pdb = PDBFile(str(test_fixed_pdb_file))
    if pdb.topology.getPeriodicBoxVectors() is None:
        raise RuntimeError("3gd8-fixed.pdb must define periodic box vectors.")
    forcefield = ForceField("amber14-all.xml")
    return pdb, forcefield


@pytest.fixture
def benchmark_simulation(
    benchmark_protein_inputs: tuple[PDBFile, ForceField],
) -> Simulation:
    """Construct a lightweight Simulation instance for integrator benchmarks."""

    pdb, forcefield = benchmark_protein_inputs
    system = create_system(pdb, forcefield)
    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picoseconds,
        2.0 * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(1234)
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = Simulation(pdb.topology, system, integrator, platform)
    box_vectors = system.getDefaultPeriodicBoxVectors()
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 5678)
    try:
        yield simulation
    finally:
        del simulation
        del integrator


def _synthetic_exchange_history(
    n_sweeps: int = 256, n_replicas: int = 6, seed: int = 42
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    history: list[list[int]] = []
    states = np.arange(n_replicas)
    for _ in range(n_sweeps):
        proposal = states.copy()
        rng.shuffle(proposal)
        history.append(proposal.tolist())
        states = proposal
    return history


def _synthetic_pair_counts(
    n_replicas: int, attempts: int = 400, seed: int = 7
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    rng = np.random.default_rng(seed)
    attempt_counts: dict[tuple[int, int], int] = {}
    accept_counts: dict[tuple[int, int], int] = {}
    for idx in range(n_replicas - 1):
        pair = (idx, idx + 1)
        att = attempts + rng.integers(-attempts // 5, attempts // 5 + 1)
        acc = max(1, int(att * rng.uniform(0.2, 0.6)))
        attempt_counts[pair] = att
        accept_counts[pair] = acc
    return attempt_counts, accept_counts


def test_exchange_statistics_benchmark(benchmark):
    """Benchmark computation of exchange statistics from synthetic history."""
    from pmarlo.replica_exchange.diagnostics import compute_exchange_statistics

    history = _synthetic_exchange_history(n_sweeps=384, n_replicas=5)
    pair_attempts, pair_accepts = _synthetic_pair_counts(len(history[0]))

    def _compute():
        return compute_exchange_statistics(
            history, len(history[0]), pair_attempts, pair_accepts
        )

    stats = benchmark(_compute)
    assert "per_pair_acceptance" in stats
    assert "replica_state_probabilities" in stats
    assert stats["average_round_trip_time"] >= 0.0


def test_exchange_statistics_large_history(benchmark):
    """Benchmark statistics computation on a longer synthetic trajectory."""
    from pmarlo.replica_exchange.diagnostics import compute_exchange_statistics

    history = _synthetic_exchange_history(n_sweeps=1024, n_replicas=7, seed=99)
    pair_attempts, pair_accepts = _synthetic_pair_counts(len(history[0]), attempts=600)

    def _compute():
        return compute_exchange_statistics(
            history, len(history[0]), pair_attempts, pair_accepts
        )

    stats = benchmark(_compute)
    assert stats["replica_state_probabilities"]


def test_diffusion_metrics_benchmark(benchmark):
    """Benchmark replica diffusion metric computation."""
    from pmarlo.replica_exchange.diagnostics import compute_diffusion_metrics

    history = _synthetic_exchange_history(n_sweeps=512, n_replicas=6, seed=21)

    def _compute():
        return compute_diffusion_metrics(history, exchange_frequency_steps=25)

    metrics = benchmark(_compute)
    assert metrics["mean_abs_disp_per_sweep"] >= 0.0
    assert len(metrics["sparkline"]) > 0


def test_temperature_ladder_retune_benchmark(benchmark, tmp_path: Path):
    """Benchmark temperature retuning optimizer."""
    from pmarlo.replica_exchange.diagnostics import retune_temperature_ladder

    temps = [300.0, 305.0, 312.0, 320.0, 335.0, 350.0]
    pair_attempts, pair_accepts = _synthetic_pair_counts(len(temps), attempts=500)
    output_json = tmp_path / "suggested_temps.json"

    def _retune():
        return retune_temperature_ladder(
            temps,
            pair_attempts,
            pair_accepts,
            target_acceptance=0.30,
            output_json=str(output_json),
            dry_run=True,
        )

    result = benchmark(_retune)
    assert output_json.exists()
    assert "suggested_temperatures" in result
    assert len(result["suggested_temperatures"]) >= 2


def test_openmm_system_builder_benchmark(benchmark, benchmark_protein_inputs):
    """Benchmark the cost of parameterizing an OpenMM System for REMD."""

    pdb, forcefield = benchmark_protein_inputs

    def _build():
        return create_system(pdb, forcefield)

    system = benchmark(_build)
    assert system.getNumParticles() == pdb.topology.getNumAtoms()
    assert system.getNumForces() >= 1


def test_langevin_integrator_single_step(benchmark, benchmark_simulation: Simulation):
    """Benchmark a single Langevin integration step within the REMD simulation."""

    simulation = benchmark_simulation

    def _step():
        simulation.step(1)

    benchmark(_step)
    state = simulation.context.getState()
    assert state.getTime().value_in_unit(unit.femtoseconds) > 0.0
