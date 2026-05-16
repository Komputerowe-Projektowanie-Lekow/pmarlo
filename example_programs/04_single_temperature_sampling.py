#!/usr/bin/env python3
"""
Single-Temperature MD for Data Collection

This example demonstrates how to use the new single-temperature MD API
to collect extensive statistics at your analysis temperature without
the overhead of replica exchange.

Strategy: When your MSM is data-starved (few frames, over-discretized states),
running multiple independent single-temperature simulations with different
random seeds is often more effective than tweaking REMD parameters.

Benefits:
- More compute goes to the target temperature
- No demultiplexing needed
- Easier to parallelize (run multiple jobs)
- Better for building extensive statistics
- Simpler analysis workflow

This is particularly useful when:
1. You have ~10k frames but 200+ MSM states (needs more data)
2. REMD acceptance is unrealistic (100% = bug)
3. You need decorrelated samples quickly
4. You want to scale horizontally (many independent runs)
"""

from pathlib import Path

from _example_support import assets_path, ensure_src_on_path, example_output_dir

ensure_src_on_path()

from pmarlo.api import emit_shards_rg_rmsd_windowed, run_single_temperature_md

TESTS_DIR = assets_path()
OUTPUT_ROOT = example_output_dir("04_single_temperature_sampling")


def run_single_trajectory_scout(seed: int = 0) -> Path:
    """Run a single independent MD simulation for data collection.

    Parameters
    ----------
    seed : int
        Random seed for this independent run

    Returns
    -------
    Path
        Path to the trajectory file
    """
    print(f"\n{'='*60}")
    print(f"Running single-T MD scout with seed {seed}")
    print(f"{'='*60}\n")

    pdb_file = TESTS_DIR / "3gd8.pdb"
    output_dir = OUTPUT_ROOT / f"run_seed_{seed:03d}"

    # Run single-temperature MD
    # Adjust these parameters based on your needs:
    # - temperature: your analysis temperature (typically 300K)
    # - total_steps: aim for 50k-100k steps per run
    # - target_frames: want 50-100 transition pairs per MSM state
    traj_files, temperature = run_single_temperature_md(
        pdb_file=str(pdb_file),
        output_dir=str(output_dir),
        temperature=300.0,  # Your analysis temperature
        total_steps=10_000,  # Short for demo; use 50k-100k in production
        random_seed=seed,
        target_frames=1000,  # Adjust based on desired time resolution
        quick=False,
    )

    print(f"\nCompleted run {seed}:")
    print(f"  Temperature: {temperature}K")
    print(f"  Trajectory files: {len(traj_files)}")
    print(f"  Output: {output_dir}")

    return Path(traj_files[0])


def run_ensemble_strategy(n_runs: int = 5) -> None:
    """Run an ensemble of independent single-T MD simulations.

    This is the recommended approach for building up data:
    1. Run 10-20 independent simulations with different seeds
    2. Each run: 50k-100k steps at your analysis temperature
    3. Combine trajectories -> emit shards -> build MSM

    Parameters
    ----------
    n_runs : int
        Number of independent runs to execute
    """
    print("\n" + "=" * 60)
    print("SINGLE-TEMPERATURE MD ENSEMBLE STRATEGY")
    print("=" * 60)
    print(f"\nRunning {n_runs} independent simulations...")
    print("Each with a different random seed for decorrelated sampling\n")

    trajectory_files = []
    for seed in range(n_runs):
        traj_file = run_single_trajectory_scout(seed=seed)
        trajectory_files.append(traj_file)

    print(f"\n{'='*60}")
    print("ENSEMBLE COMPLETE")
    print(f"{'='*60}")
    print(f"\nCollected {len(trajectory_files)} trajectories")
    print(
        f"Total frames: ~{len(trajectory_files) * 1000} (assuming 1k frames/trajectory)"
    )

    # Now emit shards from all trajectories
    print(f"\n{'='*60}")
    print("EMITTING SHARDS")
    print(f"{'='*60}\n")

    pdb_file = TESTS_DIR / "3gd8.pdb"
    shards_dir = OUTPUT_ROOT / "shards"

    shard_paths = emit_shards_rg_rmsd_windowed(
        pdb_file=str(pdb_file),
        traj_files=[str(f) for f in trajectory_files],
        out_dir=str(shards_dir),
        stride=1,  # No further subsampling
        temperature=300.0,
        seed_start=0,
    )

    print(f"\nShards created: {len(shard_paths)}")
    print(f"Shards directory: {shards_dir}")
    print("\nNext steps:")
    print("1. Use these shards to build your MSM")
    print("2. Check that you have 50-100 transitions per state")
    print("3. If still data-starved, run more independent simulations")
    print("4. Consider reducing the number of MSM states (e.g., 50 instead of 200)")


def compare_strategies():
    """Compare REMD vs single-T strategies."""
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)

    print("\nREMD with 3 replicas (300K, 310K, 320K):")
    print("  ✓ Explores temperature space")
    print("  ✓ Can escape local minima")
    print("  ✗ Only 1/3 of compute at target temperature")
    print("  ✗ Needs demultiplexing")
    print("  ✗ Complex acceptance tuning")
    print("  ✗ Harder to scale")

    print("\nSingle-T ensemble (10 runs at 300K):")
    print("  ✓ All compute at target temperature")
    print("  ✓ No demultiplexing needed")
    print("  ✓ Trivially parallel")
    print("  ✓ Simple to analyze")
    print("  ✓ Easy to add more data (just run more seeds)")
    print("  ✗ Less temperature-based exploration")

    print("\nRECOMMENDATION:")
    print("  When data-starved (few frames, many states):")
    print("  → Use single-T ensemble strategy first")
    print("  → Get 10-20 runs × 50k-100k steps")
    print("  → Build MSM with adequate statistics")
    print("  → Then consider REMD for enhanced sampling")


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("SINGLE-TEMPERATURE MD STRATEGY DEMONSTRATION")
    print("=" * 80)
    print("\nGoal: Collect decorrelated data at your analysis temperature")
    print("Use case: MSM is data-starved (few frames, over-discretized)")
    print("=" * 80)

    # Show strategy comparison
    compare_strategies()

    # Run a small ensemble (use more runs in production)
    print("\n" + "=" * 80)
    print("Running demonstration with 3 trajectories...")
    print("(In production, use 10-20 runs with 50k-100k steps each)")
    print("=" * 80)

    run_ensemble_strategy(n_runs=3)

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nNotebook/API use:")
    print("  call run_single_temperature_md(...) for each seed")
    print("  then emit_shards_rg_rmsd_windowed(...) to build shard inputs")


if __name__ == "__main__":
    main()
