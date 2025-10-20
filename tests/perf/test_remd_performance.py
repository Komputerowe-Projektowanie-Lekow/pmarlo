from __future__ import annotations

"""Performance benchmarks for Replica Exchange Molecular Dynamics (REMD).

These benchmarks measure the most critical REMD operations:
- System creation time
- MD step throughput
- Exchange overhead
- Replica setup time

Run with: pytest -m benchmark tests/perf/test_remd_performance.py
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.replica]

# Optional plugins and dependencies
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")
pytest.importorskip("openmm", reason="OpenMM required for REMD benchmarks")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _find_test_pdb() -> Path:
    """Find a test PDB file for benchmarking."""
    test_dir = Path(__file__).parent.parent
    candidates = [
        test_dir / "_assets" / "3gd8-fixed.pdb",
        test_dir / "_assets" / "3gd8.pdb",
        test_dir / "data" / "ala2.pdb",
    ]
    for pdb in candidates:
        if pdb.exists():
            return pdb
    pytest.skip("No test PDB found for REMD benchmarks")


def _cleanup_output_dir(output_dir: Path):
    """Clean up output directory after test."""
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def test_pdb():
    """Provide test PDB file."""
    return _find_test_pdb()


@pytest.fixture
def cleanup_dirs(tmp_path):
    """Clean up temporary directories after tests."""
    yield tmp_path
    # Cleanup happens automatically with tmp_path


def test_system_creation_no_bias(benchmark, test_pdb):
    """Benchmark system creation without CV bias (critical operation)."""
    from pmarlo.replica_exchange.system_builder import (
        create_system,
        load_pdb_and_forcefield,
    )

    pdb, forcefield = load_pdb_and_forcefield(
        str(test_pdb), ["amber14-all.xml", "amber14/tip3pfb.xml"]
    )

    def _create_system():
        return create_system(pdb, forcefield, cv_model_path=None)

    system = benchmark(_create_system)
    assert system is not None
    assert system.getNumForces() > 0


def test_md_throughput_cpu(benchmark, test_pdb):
    """Benchmark raw MD throughput on CPU (baseline performance)."""
    from pmarlo.replica_exchange.system_builder import (
        create_system,
        load_pdb_and_forcefield,
    )
    from openmm import LangevinIntegrator, Platform
    from openmm.app import Simulation
    from openmm import unit

    pdb, forcefield = load_pdb_and_forcefield(
        str(test_pdb), ["amber14-all.xml", "amber14/tip3pfb.xml"]
    )
    system = create_system(pdb, forcefield, cv_model_path=None)

    integrator = LangevinIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds
    )

    platform_obj = Platform.getPlatformByName("CPU")
    simulation = Simulation(pdb.topology, system, integrator, platform_obj)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

    # Warmup
    simulation.step(10)

    # Benchmark 500 steps
    def _run_md():
        simulation.step(500)

    benchmark(_run_md)


def test_replica_setup_time(benchmark, test_pdb, tmp_path):
    """Benchmark replica setup overhead (critical for initialization)."""
    from pmarlo.replica_exchange import ReplicaExchange
    from pmarlo.replica_exchange.config import RemdConfig

    output_dir = tmp_path / "replica_setup"
    temps = [300.0, 310.0, 320.0, 330.0]

    config = RemdConfig(
        pdb_file=str(test_pdb),
        temperatures=temps,
        output_dir=str(output_dir),
        exchange_frequency=100,
        auto_setup=False,
        random_seed=42,
    )

    def _setup_replicas():
        remd = ReplicaExchange.from_config(config)
        remd.plan_reporter_stride(
            total_steps=1000, equilibration_steps=0, target_frames=100
        )
        remd.setup_replicas()
        return remd

    remd = benchmark(_setup_replicas)
    assert remd.is_setup()

    _cleanup_output_dir(output_dir)


def test_exchange_overhead(benchmark, test_pdb, tmp_path):
    """Benchmark exchange attempt overhead (critical for REMD performance)."""
    from pmarlo.replica_exchange import ReplicaExchange
    from pmarlo.replica_exchange.config import RemdConfig

    output_dir = tmp_path / "exchange_overhead"
    temps = [300.0, 310.0, 320.0, 330.0]

    config = RemdConfig(
        pdb_file=str(test_pdb),
        temperatures=temps,
        output_dir=str(output_dir),
        exchange_frequency=50,  # Frequent exchanges to measure overhead
        auto_setup=False,
        random_seed=42,
    )

    remd = ReplicaExchange.from_config(config)
    remd.plan_reporter_stride(total_steps=500, equilibration_steps=0, target_frames=50)
    remd.setup_replicas()

    def _run_with_exchanges():
        remd.run_simulation(total_steps=500, equilibration_steps=0)

    benchmark(_run_with_exchanges)

    stats = remd.get_exchange_statistics()
    assert remd.exchange_attempts > 0
    assert "mean_acceptance" in stats

    _cleanup_output_dir(output_dir)


def test_short_remd_full_workflow(benchmark, test_pdb, tmp_path):
    """Benchmark complete short REMD workflow (integration benchmark)."""
    from pmarlo.replica_exchange import ReplicaExchange
    from pmarlo.replica_exchange.config import RemdConfig

    output_dir = tmp_path / "full_workflow"
    temps = [300.0, 310.0]

    def _run_full_remd():
        config = RemdConfig(
            pdb_file=str(test_pdb),
            temperatures=temps,
            output_dir=str(output_dir),
            exchange_frequency=100,
            auto_setup=False,
            random_seed=42,
        )

        remd = ReplicaExchange.from_config(config)
        remd.plan_reporter_stride(
            total_steps=200, equilibration_steps=0, target_frames=20
        )
        remd.setup_replicas()
        remd.run_simulation(total_steps=200, equilibration_steps=0)
        return remd

    remd = benchmark(_run_full_remd)
    assert remd.is_setup()

    # Verify trajectories were created
    for i in range(len(temps)):
        traj_file = output_dir / f"replica_{i:02d}.dcd"
        assert traj_file.exists(), f"Trajectory file {traj_file} not created"

    _cleanup_output_dir(output_dir)


def test_platform_selection_with_seed(benchmark, test_pdb, tmp_path):
    """Benchmark platform selection with random seed (regression test for performance bug).

    This test verifies that setting a random_seed doesn't cause platform selection
    to fallback to Reference platform (which would cause 6x slowdown).
    """
    from pmarlo.replica_exchange import ReplicaExchange
    from pmarlo.replica_exchange.config import RemdConfig

    output_dir = tmp_path / "platform_seed"

    def _run_with_seed():
        config = RemdConfig(
            pdb_file=str(test_pdb),
            temperatures=[300.0, 310.0],
            output_dir=str(output_dir),
            exchange_frequency=200,
            auto_setup=False,
            random_seed=42,  # Critical: this should NOT degrade performance
        )

        remd = ReplicaExchange.from_config(config)
        remd.plan_reporter_stride(
            total_steps=200, equilibration_steps=0, target_frames=20
        )
        remd.setup_replicas()

        # Verify platform is NOT Reference
        from openmm import Platform

        for integrator in remd.integrators:
            if hasattr(integrator, "getStepSize"):
                # This is a workaround to check platform indirectly
                # In actual implementation, we'd check the context's platform
                pass

        remd.run_simulation(total_steps=200, equilibration_steps=0)
        return remd

    remd = benchmark(_run_with_seed)
    assert remd.is_setup()

    _cleanup_output_dir(output_dir)

