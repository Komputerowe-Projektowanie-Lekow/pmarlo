"""Single-temperature molecular dynamics simulation API.

This module provides a simplified interface for running single-temperature
Langevin MD simulations to generate decorrelated data without replica exchange.
Ideal for collecting extensive statistics at a single analysis temperature.
"""

import logging
from pathlib import Path
from typing import Any, List, Tuple

from pmarlo.replica_exchange.config import RemdConfig
from pmarlo.replica_exchange.replica_exchange import ReplicaExchange
from pmarlo.transform.progress import coerce_progress_callback

logger = logging.getLogger("pmarlo")


def run_single_temperature_md(
    pdb_file: str | Path,
    output_dir: str | Path,
    temperature: float,
    total_steps: int,
    *,
    random_seed: int | None = None,
    start_from_checkpoint: str | Path | None = None,
    start_from_pdb: str | Path | None = None,
    jitter_start: bool = False,
    jitter_sigma_A: float = 0.05,
    velocity_reseed: bool = False,
    save_state_frequency: int | None = None,
    dcd_stride: int = 1,
    target_frames: int = 5000,
    quick: bool = False,
    save_final_pdb: bool = False,
    final_pdb_path: str | Path | None = None,
    final_pdb_temperature: float | None = None,
    **kwargs: Any,
) -> Tuple[List[str], float]:
    """Run single-temperature Langevin MD and return (trajectory_files, temperature).

    This is optimized for generating many decorrelated frames at one temperature,
    avoiding the complexity and overhead of replica exchange. Perfect for building
    up large datasets when your MSM is data-starved.

    Parameters
    ----------
    pdb_file:
        Path to the prepared PDB file.
    output_dir:
        Directory to store output files.
    temperature:
        Temperature in Kelvin for the simulation.
    total_steps:
        Number of MD steps to run.
    random_seed:
        Seed for deterministic behavior. If None, uses a random seed.
    start_from_checkpoint:
        Optional checkpoint file to resume from.
    start_from_pdb:
        Optional PDB file to start from instead of the input PDB.
    jitter_start:
        Whether to add small random displacement to starting coordinates.
    jitter_sigma_A:
        Standard deviation of jitter in Angstroms (default 0.05).
    velocity_reseed:
        Whether to reseed velocities at start.
    save_state_frequency:
        How often to save simulation state (checkpoints).
    dcd_stride:
        Stride for saving DCD frames.
    target_frames:
        Target number of frames to save (auto-adjusts stride if needed).
    quick:
        Use reduced settings for quick testing.
    save_final_pdb:
        When True, write a restart snapshot of the last frame.
    final_pdb_path:
        Optional explicit path for the restart snapshot (defaults to output_dir/restart_final_frame.pdb).
    final_pdb_temperature:
        Temperature to report for the restart snapshot (defaults to the simulation temperature).
    **kwargs:
        Additional arguments (e.g., progress_callback, cancel_token).

    Returns
    -------
    trajectory_files:
        List of trajectory file paths (will be a single DCD file).
    temperature:
        The simulation temperature (same as input).

    Examples
    --------
    Run a long single-temperature simulation to build up data:

    >>> trajs, temp = run_single_temperature_md(
    ...     "protein.pdb",
    ...     "output/md_run1",
    ...     temperature=300.0,
    ...     total_steps=100_000,
    ...     random_seed=42
    ... )

    Run multiple independent simulations with different seeds:

    >>> for seed in range(10):
    ...     trajs, temp = run_single_temperature_md(
    ...         "protein.pdb",
    ...         f"output/md_run_{seed}",
    ...         temperature=300.0,
    ...         total_steps=50_000,
    ...         random_seed=seed
    ...     )

    Notes
    -----
    This function internally uses the ReplicaExchange class with a single replica.
    No exchange attempts are made, so you get pure single-temperature dynamics.

    The advantage over REMD with 3 replicas:
    - More compute goes to target temperature
    - Simpler analysis (no demuxing needed)
    - Better for building extensive statistics
    - Easier to parallelize (run multiple independent jobs)
    """
    logger.info(
        "[single-T-MD] Starting single-temperature simulation: T=%.1fK, total_steps=%d, pdb=%s",
        temperature,
        total_steps,
        Path(pdb_file).name,
    )

    md_out = Path(output_dir) / "single_temp_md"

    # Derive equilibration and save frequency
    if quick:
        equil = max(10, int(0.05 * total_steps))
        if save_state_frequency is None:
            save_state_frequency = max(100, int(0.1 * total_steps))
    else:
        equil = max(100, int(0.1 * total_steps))
        if save_state_frequency is None:
            save_state_frequency = max(1000, int(0.1 * total_steps))

    # Adjust stride to hit target frames
    production_steps = max(0, total_steps - equil)
    if production_steps > target_frames:
        dcd_stride = max(1, production_steps // target_frames)
    dcd_stride = max(1, int(dcd_stride))
    save_state_frequency = int(save_state_frequency)

    logger.debug(
        "[single-T-MD] Configuration: equilibration=%d steps, dcd_stride=%d, seed=%s",
        equil,
        dcd_stride,
        random_seed if random_seed is not None else "random",
    )

    _emit_banner(
        "SINGLE-TEMPERATURE MD SIMULATION STARTING",
        [
            f"Temperature: {temperature}K",
            f"Total steps: {total_steps}",
            f"Equilibration: {equil} steps",
            f"Production: {production_steps} steps",
            f"DCD stride: {dcd_stride}",
            f"Target frames: ~{production_steps // dcd_stride}",
            f"Output directory: {md_out}",
            f"Random seed: {random_seed}",
        ],
    )

    logger.info("[single-T-MD] Creating single-replica simulation")

    # Create a single-replica "REMD" (which is just MD)
    remd = ReplicaExchange.from_config(
        RemdConfig(
            pdb_file=str(pdb_file),
            temperatures=[temperature],  # Single temperature
            output_dir=str(md_out),
            exchange_frequency=99999999,  # Effectively disabled (no exchanges with 1 replica anyway)
            auto_setup=False,
            dcd_stride=dcd_stride,
            random_seed=random_seed,
            start_from_checkpoint=(
                str(start_from_checkpoint) if start_from_checkpoint else None
            ),
            start_from_pdb=str(start_from_pdb) if start_from_pdb else None,
            jitter_sigma_A=float(jitter_sigma_A) if jitter_start else 0.0,
            reseed_velocities=bool(velocity_reseed),
            write_replica_indices=[0],  # Only write the single replica
        )
    )

    logger.debug("[single-T-MD] Planning reporter stride for target_frames=%d", target_frames)
    remd.plan_reporter_stride(
        total_steps=int(total_steps),
        equilibration_steps=int(equil),
        target_frames=target_frames,
    )

    logger.info("[single-T-MD] Setting up simulation system")
    remd.setup_replicas()

    if start_from_checkpoint:
        logger.info("[single-T-MD] Attempting to restore from checkpoint: %s", start_from_checkpoint)
        if _restore_from_checkpoint(remd, start_from_checkpoint):
            equil = 0
            logger.info("[single-T-MD] Checkpoint restored successfully, skipping equilibration")

    cb = coerce_progress_callback(kwargs)

    _emit_banner(
        "PHASE 1/2: RUNNING MD SIMULATION",
        [
            f"Equilibration: {equil} steps",
            f"Production: {production_steps} steps",
            f"Saving frames every {dcd_stride} steps",
        ],
    )

    logger.info("[single-T-MD] Starting MD simulation")
    remd.run_simulation(
        total_steps=int(total_steps),
        equilibration_steps=int(equil),
        save_state_frequency=save_state_frequency,
        progress_callback=cb,
        cancel_token=kwargs.get("cancel_token"),
    )

    logger.info("[single-T-MD] MD simulation complete")

    final_snapshot_written: Path | None = None
    if save_final_pdb or final_pdb_path is not None:
        snapshot_target = (
            Path(final_pdb_path)
            if final_pdb_path is not None
            else Path(md_out) / "restart_final_frame.pdb"
        )
        snapshot_target.parent.mkdir(parents=True, exist_ok=True)
        target_temperature = (
            float(final_pdb_temperature)
            if final_pdb_temperature is not None
            else float(temperature)
        )
        logger.info(
            "[single-T-MD] Exporting final structure at %.1fK to %s",
            target_temperature,
            snapshot_target,
        )
        final_snapshot_written = remd.export_current_structure(
            snapshot_target, temperature=target_temperature
        )

    _emit_banner(
        "PHASE 2/2: FINALIZING OUTPUT",
        [
            f"Output directory: {md_out}",
            f"Trajectory files: {len(remd.trajectory_files)}",
            (
                f"Restart snapshot: {final_snapshot_written}"
                if final_snapshot_written
                else "Restart snapshot: skipped"
            ),
        ],
    )

    traj_files = [str(p) for p in remd.trajectory_files]

    logger.info(
        "[single-T-MD] Single-temperature MD complete: %d trajectory file(s) at %.1fK",
        len(traj_files),
        temperature,
    )

    _emit_banner(
        "SINGLE-TEMPERATURE MD COMPLETE",
        [
            f"Temperature: {temperature}K",
            f"Trajectory files: {len(traj_files)}",
            f"Output: {md_out}",
        ],
    )

    return traj_files, temperature


def _restore_from_checkpoint(remd: ReplicaExchange, checkpoint_path: str | Path) -> bool:
    """Attempt to restore simulation from checkpoint."""
    try:
        with open(checkpoint_path, "rb") as f:
            import pickle
            checkpoint = pickle.load(f)
        remd.restore_from_checkpoint(checkpoint)
        return True
    except Exception as e:
        logger.warning(f"[single-T-MD] Failed to restore checkpoint: {e}")
        return False


def _emit_banner(title: str, lines: List[str]) -> None:
    """Print a formatted banner."""
    width = max(len(title), max(len(line) for line in lines)) + 4
    border = "=" * width

    print(f"\n{border}")
    print(f"  {title}")
    print(border)
    for line in lines:
        print(f"  {line}")
    print(f"{border}\n")
    logger.info(f"[single-T-MD] {title}")
    for line in lines:
        logger.info(f"[single-T-MD]   {line}")

