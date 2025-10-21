#!/usr/bin/env python
"""Benchmark REMD performance to identify bottlenecks.

This script profiles different components of REMD to find performance issues:
1. System creation time
2. Replica setup time
3. MD step throughput (steps/sec)
4. Exchange overhead
5. CV monitoring overhead (if enabled)
6. Reporter/checkpoint overhead

Usage:
    python benchmark_remd_performance.py

    # With CV model:
    python benchmark_remd_performance.py --cv-model path/to/model.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_test_pdb() -> Path:
    """Find a test PDB file."""
    candidates = [
        Path(__file__).parent.parent / "tests" / "_assets" / "3gd8-fixed.pdb",
        Path(__file__).parent.parent / "tests" / "_assets" / "3gd8.pdb",
    ]
    for pdb in candidates:
        if pdb.exists():
            return pdb
    raise FileNotFoundError("No test PDB found. Checked: " + ", ".join(str(c) for c in candidates))

def benchmark_system_creation(pdb_file: Path, cv_model: Path | None) -> Dict[str, Any]:
    """Benchmark system creation time."""
    logger.info("=" * 80)
    logger.info("BENCHMARK: System Creation")
    logger.info("=" * 80)

    from pmarlo.replica_exchange.system_builder import create_system, load_pdb_and_forcefield
    from openmm.app import ForceField

    results = {}

    # Load PDB and forcefield
    start = time.perf_counter()
    pdb, forcefield = load_pdb_and_forcefield(
        str(pdb_file),
        ["amber14-all.xml", "amber14/tip3pfb.xml"]
    )
    load_time = time.perf_counter() - start
    results['pdb_load_time'] = load_time
    logger.info(f"PDB + Forcefield load: {load_time:.3f}s")

    # Create system WITHOUT CV bias
    start = time.perf_counter()
    system_no_bias = create_system(pdb, forcefield, cv_model_path=None)
    sys_time_no_bias = time.perf_counter() - start
    results['system_creation_no_bias'] = sys_time_no_bias
    logger.info(f"System creation (no bias): {sys_time_no_bias:.3f}s")
    logger.info(f"  Forces in system: {system_no_bias.getNumForces()}")

    # Create system WITH CV bias (if model provided)
    if cv_model:
        start = time.perf_counter()
        try:
            system_with_bias = create_system(pdb, forcefield, cv_model_path=str(cv_model))
            sys_time_with_bias = time.perf_counter() - start
            results['system_creation_with_bias'] = sys_time_with_bias
            logger.info(f"System creation (with bias): {sys_time_with_bias:.3f}s")
            logger.info(f"  Forces in system: {system_with_bias.getNumForces()}")
            logger.info(f"  Overhead: {(sys_time_with_bias - sys_time_no_bias) * 1000:.1f}ms")
        except Exception as e:
            logger.error(f"Failed to create system with CV bias: {e}")
            results['system_creation_with_bias'] = None

    return results

def benchmark_md_throughput(pdb_file: Path, steps: int = 1000, platform: str = "CPU") -> Dict[str, Any]:
    """Benchmark raw MD throughput without exchanges."""
    logger.info("=" * 80)
    logger.info(f"BENCHMARK: MD Throughput ({steps} steps, {platform})")
    logger.info("=" * 80)

    from pmarlo.replica_exchange.system_builder import create_system, load_pdb_and_forcefield
    from openmm import LangevinIntegrator, Platform
    from openmm.app import Simulation
    from openmm import unit

    pdb, forcefield = load_pdb_and_forcefield(
        str(pdb_file),
        ["amber14-all.xml", "amber14/tip3pfb.xml"]
    )
    system = create_system(pdb, forcefield, cv_model_path=None)

    integrator = LangevinIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtoseconds
    )

    platform_obj = Platform.getPlatformByName(platform)
    simulation = Simulation(pdb.topology, system, integrator, platform_obj)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

    # Warmup
    simulation.step(10)

    # Benchmark
    start = time.perf_counter()
    simulation.step(steps)
    elapsed = time.perf_counter() - start

    throughput = steps / elapsed
    logger.info(f"MD throughput: {throughput:.1f} steps/sec")
    logger.info(f"Time per step: {elapsed / steps * 1000:.3f}ms")
    logger.info(f"Time per ps: {elapsed / (steps * 0.002):.3f}s")

    return {
        'md_steps_per_sec': throughput,
        'time_per_step_ms': elapsed / steps * 1000,
        'total_time': elapsed,
    }

def benchmark_exchange_overhead(pdb_file: Path, n_replicas: int = 4) -> Dict[str, Any]:
    """Benchmark exchange attempt overhead."""
    logger.info("=" * 80)
    logger.info(f"BENCHMARK: Exchange Overhead ({n_replicas} replicas)")
    logger.info("=" * 80)

    from pmarlo.replica_exchange import ReplicaExchange
    from pmarlo.replica_exchange.config import RemdConfig

    temps = [300.0 + i * 10.0 for i in range(n_replicas)]

    config = RemdConfig(
        pdb_file=str(pdb_file),
        temperatures=temps,
        output_dir="tmp_benchmark",
        exchange_frequency=50,  # Frequent exchanges to measure overhead
        auto_setup=False,
        random_seed=42,
    )

    remd = ReplicaExchange.from_config(config)

    # Setup
    setup_start = time.perf_counter()
    remd.plan_reporter_stride(total_steps=1000, equilibration_steps=0, target_frames=100)
    remd.setup_replicas()
    setup_time = time.perf_counter() - setup_start

    logger.info(f"Replica setup time: {setup_time:.3f}s ({setup_time/n_replicas:.3f}s per replica)")

    # Run short simulation with exchanges
    run_start = time.perf_counter()
    remd.run_simulation(total_steps=500, equilibration_steps=0)
    run_time = time.perf_counter() - run_start

    stats = remd.get_exchange_statistics()
    n_exchanges = remd.exchange_attempts

    logger.info(f"Run time (500 steps): {run_time:.3f}s")
    logger.info(f"Throughput: {500 * n_replicas / run_time:.1f} total steps/sec")
    logger.info(f"Exchange attempts: {n_exchanges}")
    logger.info(f"Exchange acceptance: {stats.get('mean_acceptance', 0.0):.2%}")

    if n_exchanges > 0:
        time_per_exchange = run_time / n_exchanges
        logger.info(f"Time per exchange attempt: {time_per_exchange * 1000:.2f}ms")

    # Cleanup
    import shutil
    if Path("tmp_benchmark").exists():
        shutil.rmtree("tmp_benchmark")

    return {
        'setup_time': setup_time,
        'setup_time_per_replica': setup_time / n_replicas,
        'run_time': run_time,
        'throughput_total_steps_per_sec': 500 * n_replicas / run_time,
        'n_exchanges': n_exchanges,
        'acceptance': stats.get('mean_acceptance', 0.0),
    }

def benchmark_frequency_comparison(pdb_file: Path) -> Dict[str, Any]:
    """Compare performance with different exchange frequencies."""
    logger.info("=" * 80)
    logger.info("BENCHMARK: Exchange Frequency Impact")
    logger.info("=" * 80)

    from pmarlo.replica_exchange import ReplicaExchange
    from pmarlo.replica_exchange.config import RemdConfig
    import shutil

    temps = [300.0, 310.0, 320.0, 330.0]
    frequencies = [10, 50, 100, 250, 500]
    results = {}

    for freq in frequencies:
        config = RemdConfig(
            pdb_file=str(pdb_file),
            temperatures=temps,
            output_dir=f"tmp_bench_freq_{freq}",
            exchange_frequency=freq,
            auto_setup=False,
            random_seed=42,
        )

        remd = ReplicaExchange.from_config(config)
        remd.plan_reporter_stride(total_steps=1000, equilibration_steps=0, target_frames=100)
        remd.setup_replicas()

        start = time.perf_counter()
        remd.run_simulation(total_steps=1000, equilibration_steps=0)
        elapsed = time.perf_counter() - start

        throughput = 1000 * len(temps) / elapsed
        n_exchanges = remd.exchange_attempts

        logger.info(f"Frequency {freq:4d}: {elapsed:.2f}s, {throughput:.0f} steps/s, {n_exchanges:2d} exchanges")

        results[freq] = {
            'time': elapsed,
            'throughput': throughput,
            'n_exchanges': n_exchanges,
        }

        # Cleanup
        if Path(f"tmp_bench_freq_{freq}").exists():
            shutil.rmtree(f"tmp_bench_freq_{freq}")

    # Find optimal
    best_freq = max(results.items(), key=lambda x: x[1]['throughput'])
    logger.info(f"\nOptimal frequency: {best_freq[0]} steps (throughput: {best_freq[1]['throughput']:.0f} steps/s)")

    return results

def benchmark_reporter_overhead(pdb_file: Path) -> Dict[str, Any]:
    """Benchmark DCD reporter overhead."""
    logger.info("=" * 80)
    logger.info("BENCHMARK: Reporter Overhead")
    logger.info("=" * 80)

    from pmarlo.replica_exchange import ReplicaExchange
    from pmarlo.replica_exchange.config import RemdConfig
    import shutil

    temps = [300.0, 310.0, 320.0, 330.0]
    results = {}

    # Run WITHOUT reporter (stride = very high)
    config = RemdConfig(
        pdb_file=str(pdb_file),
        temperatures=temps,
        output_dir="tmp_no_reporter",
        exchange_frequency=200,
        dcd_stride=1000000,  # Effectively disabled
        auto_setup=False,
        random_seed=42,
    )

    remd = ReplicaExchange.from_config(config)
    remd.plan_reporter_stride(total_steps=1000, equilibration_steps=0, target_frames=1)  # Minimal
    remd.setup_replicas()

    start = time.perf_counter()
    remd.run_simulation(total_steps=1000, equilibration_steps=0)
    time_no_reporter = time.perf_counter() - start

    logger.info(f"Without reporter: {time_no_reporter:.3f}s")

    if Path("tmp_no_reporter").exists():
        shutil.rmtree("tmp_no_reporter")

    # Run WITH reporter (stride = 1)
    config = RemdConfig(
        pdb_file=str(pdb_file),
        temperatures=temps,
        output_dir="tmp_with_reporter",
        exchange_frequency=200,
        dcd_stride=1,  # Every step
        auto_setup=False,
        random_seed=42,
    )

    remd = ReplicaExchange.from_config(config)
    remd.plan_reporter_stride(total_steps=1000, equilibration_steps=0, target_frames=1000)
    remd.setup_replicas()

    start = time.perf_counter()
    remd.run_simulation(total_steps=1000, equilibration_steps=0)
    time_with_reporter = time.perf_counter() - start

    logger.info(f"With reporter (stride=1): {time_with_reporter:.3f}s")
    logger.info(f"Overhead: {(time_with_reporter - time_no_reporter) / time_no_reporter * 100:.1f}%")

    if Path("tmp_with_reporter").exists():
        shutil.rmtree("tmp_with_reporter")

    return {
        'no_reporter_time': time_no_reporter,
        'with_reporter_time': time_with_reporter,
        'overhead_pct': (time_with_reporter - time_no_reporter) / time_no_reporter * 100,
    }

def print_summary(results: Dict[str, Any]):
    """Print benchmark summary."""
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    if 'system_creation' in results:
        logger.info("\nSystem Creation:")
        logger.info(f"  No bias:   {results['system_creation'].get('system_creation_no_bias', 0):.3f}s")
        if results['system_creation'].get('system_creation_with_bias'):
            logger.info(f"  With bias: {results['system_creation']['system_creation_with_bias']:.3f}s")

    if 'md_throughput' in results:
        logger.info("\nMD Throughput (single replica, no exchanges):")
        logger.info(f"  {results['md_throughput']['md_steps_per_sec']:.1f} steps/sec")
        logger.info(f"  {results['md_throughput']['time_per_step_ms']:.3f}ms per step")

    if 'exchange_overhead' in results:
        logger.info("\nREMD with Exchanges:")
        logger.info(f"  Setup: {results['exchange_overhead']['setup_time']:.3f}s")
        logger.info(f"  Throughput: {results['exchange_overhead']['throughput_total_steps_per_sec']:.1f} total steps/sec")
        logger.info(f"  Exchanges: {results['exchange_overhead']['n_exchanges']}")

    if 'frequency_comparison' in results:
        logger.info("\nExchange Frequency Impact:")
        for freq, data in sorted(results['frequency_comparison'].items()):
            logger.info(f"  {freq:4d} steps: {data['throughput']:.0f} steps/s ({data['n_exchanges']:2d} exchanges)")

    if 'reporter_overhead' in results:
        logger.info("\nReporter Overhead:")
        logger.info(f"  Without: {results['reporter_overhead']['no_reporter_time']:.3f}s")
        logger.info(f"  With:    {results['reporter_overhead']['with_reporter_time']:.3f}s")
        logger.info(f"  Overhead: {results['reporter_overhead']['overhead_pct']:.1f}%")

    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)

    # Analyze and provide recommendations
    if 'frequency_comparison' in results:
        freqs = results['frequency_comparison']
        best = max(freqs.items(), key=lambda x: x[1]['throughput'])
        logger.info(f"\n1. EXCHANGE FREQUENCY: Use {best[0]} steps for optimal performance")
        logger.info(f"   (Current default with quick=True is 50 steps, which may be too frequent)")

    if 'reporter_overhead' in results:
        overhead = results['reporter_overhead']['overhead_pct']
        if overhead > 10:
            logger.info(f"\n2. REPORTER STRIDE: Current overhead is {overhead:.1f}%")
            logger.info("   Consider reducing reporter frequency if not needed")

    if 'md_throughput' in results and 'exchange_overhead' in results:
        single_throughput = results['md_throughput']['md_steps_per_sec']
        remd_throughput = results['exchange_overhead']['throughput_total_steps_per_sec']
        n_replicas = 4  # From benchmark
        expected = single_throughput * n_replicas
        efficiency = remd_throughput / expected * 100
        logger.info(f"\n3. REMD EFFICIENCY: {efficiency:.1f}%")
        logger.info(f"   Expected: {expected:.0f} steps/s (single replica × {n_replicas})")
        logger.info(f"   Actual:   {remd_throughput:.0f} steps/s")
        if efficiency < 70:
            logger.info("   ⚠️  LOW EFFICIENCY - investigate exchange/reporter overhead")

def main():
    parser = argparse.ArgumentParser(description="Benchmark REMD performance")
    parser.add_argument("--pdb", type=Path, help="PDB file (default: auto-detect test file)")
    parser.add_argument("--cv-model", type=Path, help="CV model for bias benchmarking")
    parser.add_argument("--platform", default="CPU", help="OpenMM platform (CPU/CUDA)")
    parser.add_argument("--steps", type=int, default=1000, help="Steps for MD throughput benchmark")
    parser.add_argument("--skip-system", action="store_true", help="Skip system creation benchmark")
    parser.add_argument("--skip-throughput", action="store_true", help="Skip MD throughput benchmark")
    parser.add_argument("--skip-exchange", action="store_true", help="Skip exchange overhead benchmark")
    parser.add_argument("--skip-frequency", action="store_true", help="Skip frequency comparison")
    parser.add_argument("--skip-reporter", action="store_true", help="Skip reporter overhead benchmark")

    args = parser.parse_args()

    # Find PDB
    pdb_file = args.pdb or find_test_pdb()
    logger.info(f"Using PDB: {pdb_file}")

    results = {}

    # Run benchmarks
    if not args.skip_system:
        results['system_creation'] = benchmark_system_creation(pdb_file, args.cv_model)

    if not args.skip_throughput:
        results['md_throughput'] = benchmark_md_throughput(pdb_file, args.steps, args.platform)

    if not args.skip_exchange:
        results['exchange_overhead'] = benchmark_exchange_overhead(pdb_file)

    if not args.skip_frequency:
        results['frequency_comparison'] = benchmark_frequency_comparison(pdb_file)

    if not args.skip_reporter:
        results['reporter_overhead'] = benchmark_reporter_overhead(pdb_file)

    # Print summary
    print_summary(results)

    # Save results
    import json
    output_file = Path("benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
