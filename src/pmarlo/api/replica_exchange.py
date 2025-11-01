import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Protocol, Tuple

import numpy as np

from pmarlo.io.trajectory_reader import MDTrajReader
from pmarlo.replica_exchange.config import RemdConfig
from pmarlo.replica_exchange.replica_exchange import ReplicaExchange
from pmarlo.transform.progress import coerce_progress_callback

logger = logging.getLogger("pmarlo")


class ReplicaExchangeProtocol(Protocol):
    """Protocol for replica exchange simulation objects."""
    cv_model_path: str | None
    cv_scaler_mean: Any | None
    cv_scaler_scale: Any | None
    reporter_stride: int | None
    dcd_stride: int | None
    output_dir: str
    trajectory_files: list[Path]

    def restore_from_checkpoint(self, checkpoint: Any) -> None: ...
    def plan_reporter_stride(self, total_steps: int, equilibration_steps: int, target_frames: int) -> None: ...
    def setup_replicas(self) -> None: ...
    def run_simulation(
        self,
        total_steps: int,
        equilibration_steps: int,
        save_state_frequency: int,
        progress_callback: Any,
        cancel_token: Any,
    ) -> None: ...
    def export_current_structure(self, path: Path, temperature: float) -> Path: ...
    def demux_trajectories(
        self,
        target_temperature: float,
        equilibration_steps: int,
        progress_callback: Any,
    ) -> str | Path | None: ...


def run_replica_exchange(
    pdb_file: str | Path,
    output_dir: str | Path,
    temperatures: List[float],
    total_steps: int,
    *,
    random_seed: int | None = None,
    random_state: int | None = None,
    start_from_checkpoint: str | Path | None = None,
    start_from_pdb: str | Path | None = None,
    cv_model_path: str | Path | None = None,
    cv_scaler_mean: Any | None = None,
    cv_scaler_scale: Any | None = None,
    jitter_start: bool = False,
    jitter_sigma_A: float = 0.05,
    velocity_reseed: bool = False,
    exchange_frequency_steps: int | None = None,
    save_state_frequency: int | None = None,
    temperature_schedule_mode: str | None = None,
    save_final_pdb: bool = False,
    final_pdb_path: str | Path | None = None,
    final_pdb_temperature: float | None = None,
    **kwargs: Any,
) -> Tuple[List[str], List[float]]:
    """Run REMD and return (trajectory_files, analysis_temperatures).

    Attempts demultiplexing to ~300 K; falls back to per-replica trajectories.
    When ``random_state`` or ``random_seed`` is provided, the seed is forwarded
    to the underlying :class:`ReplicaExchange` for deterministic behavior. If
    both are provided, ``random_state`` takes precedence for backward
    compatibility.
    """
    logger.info(
        "[remd] Starting replica exchange: n_replicas=%d, total_steps=%d, pdb=%s",
        len(temperatures),
        total_steps,
        Path(pdb_file).name,
    )

    remd_out = Path(output_dir) / "replica_exchange"

    quick_mode = bool(kwargs.get("quick", False))
    equil, exchange_frequency, dcd_stride = _derive_run_plan(
        total_steps, quick_mode, exchange_frequency_steps
    )
    seed = _resolve_simulation_seed(random_seed, random_state)

    logger.debug(
        "[remd] Configuration: equilibration=%d steps, exchange_freq=%d steps, dcd_stride=%d, seed=%s",
        equil,
        exchange_frequency,
        dcd_stride,
        seed if seed is not None else "random",
    )

    _emit_banner(
        "REPLICA EXCHANGE SIMULATION STARTING",
        [
            f"Number of replicas: {len(temperatures)}",
            f"Temperature ladder: {temperatures}",
            f"Total steps: {total_steps}",
            f"Output directory: {remd_out}",
            f"Random seed: {seed}",
        ],
    )

    logger.info("[remd] Creating ReplicaExchange instance from config")
    remd = ReplicaExchange.from_config(
        RemdConfig(
            pdb_file=str(pdb_file),
            temperatures=temperatures,
            output_dir=str(remd_out),
            exchange_frequency=exchange_frequency,
            auto_setup=False,
            dcd_stride=dcd_stride,
            random_seed=seed,
            start_from_checkpoint=(
                str(start_from_checkpoint) if start_from_checkpoint else None
            ),
            start_from_pdb=str(start_from_pdb) if start_from_pdb else None,
            jitter_sigma_A=float(jitter_sigma_A) if jitter_start else 0.0,
            reseed_velocities=bool(velocity_reseed),
            temperature_schedule_mode=temperature_schedule_mode,
        )
    )
    _configure_cv_model(remd, cv_model_path, cv_scaler_mean, cv_scaler_scale)

    logger.debug("[remd] Planning reporter stride for target_frames=5000")
    remd.plan_reporter_stride(
        total_steps=int(total_steps), equilibration_steps=int(equil), target_frames=5000
    )

    logger.info("[remd] Setting up %d replica systems", len(temperatures))
    remd.setup_replicas()

    if start_from_checkpoint:
        logger.info("[remd] Attempting to restore from checkpoint: %s", start_from_checkpoint)
        if _restore_from_checkpoint(remd, start_from_checkpoint):
            equil = 0
            logger.info("[remd] Checkpoint restored successfully, skipping equilibration")

    cb = coerce_progress_callback(kwargs)

    _emit_banner(
        "PHASE 1/2: RUNNING MD SIMULATION",
        [
            f"This will run {len(temperatures)} parallel replicas",
            f"Each replica will run for {total_steps} MD steps",
            f"Equilibration: {equil} steps",
            "Press Ctrl+C to cancel the simulation",
        ],
    )

    logger.info("[remd] Starting simulation: total_steps=%d, equilibration=%d", total_steps, equil)
    remd.run_simulation(
        total_steps=int(total_steps),
        equilibration_steps=int(equil),
        save_state_frequency=int(save_state_frequency or 10_000),
        progress_callback=cb,
        cancel_token=kwargs.get("cancel_token"),
    )
    logger.info("[remd] Simulation completed successfully")

    final_snapshot_written: Optional[Path] = None
    if save_final_pdb or final_pdb_path is not None:
        snapshot_target = (
            Path(final_pdb_path)
            if final_pdb_path is not None
            else Path(remd.output_dir) / "restart_final_frame.pdb"
        )
        target_temperature = (
            float(final_pdb_temperature)
            if final_pdb_temperature is not None
            else float(temperatures[0])
        )
        logger.info("[remd] Exporting final structure at T=%.1fK to %s", target_temperature, snapshot_target)
        final_snapshot_written = remd.export_current_structure(
            snapshot_target, temperature=target_temperature
        )
        _emit_banner(
            "REPLICA EXCHANGE SNAPSHOT SAVED",
            [f"Restart PDB written to {final_snapshot_written}"],
        )

    _emit_banner(
        "PHASE 1/2: MD SIMULATION COMPLETE",
        [f"Generated {len(remd.trajectory_files)} replica trajectories"],
    )

    _emit_banner(
        "PHASE 2/2: DEMULTIPLEXING TRAJECTORIES",
        [
            "Extracting frames at target temperature (300K)",
            "This creates a single trajectory from replica exchanges",
            "WARNING: This phase cannot be cancelled with Ctrl+C (runs to completion)",
        ],
    )

    logger.info("[remd] Starting demultiplexing to target_temperature=300.0K")
    demuxed = remd.demux_trajectories(
        target_temperature=300.0, equilibration_steps=int(equil), progress_callback=cb
    )

    _emit_banner("PHASE 2/2: DEMULTIPLEXING COMPLETE")

    logger.info("[remd] Evaluating demultiplexed trajectory quality")
    accepted, nframes = _evaluate_demux_result(
        remd=remd,
        demuxed_path=demuxed,
        total_steps=total_steps,
        equilibration_steps=equil,
        pdb_file=pdb_file,
    )
    if accepted and demuxed:
        logger.info("[remd] Demultiplexing successful: %d frames in trajectory", nframes)
        _emit_banner(
            "REPLICA EXCHANGE COMPLETE - SUCCESS",
            [
                f"Returning demultiplexed trajectory: {demuxed}",
                f"Total frames in demuxed trajectory: {nframes}",
            ],
        )
        return [str(demuxed)], [300.0]

    logger.warning(
        "[remd] Demultiplexing produced insufficient frames (threshold not met), falling back to per-replica trajectories"
    )
    _emit_banner(
        "REPLICA EXCHANGE COMPLETE - FALLBACK TO PER-REPLICA TRAJECTORIES",
        [
            "Demultiplexing did not produce enough frames",
            f"Returning {len(remd.trajectory_files)} per-replica trajectories instead",
        ],
    )
    traj_files = [str(f) for f in remd.trajectory_files]
    logger.info("[remd] Returning %d per-replica trajectories", len(traj_files))
    return traj_files, temperatures

def _resolve_simulation_seed(
    random_seed: int | None,
    random_state: int | None,
) -> int | None:
    """Resolve a deterministic simulation seed preferring ``random_state``."""

    if random_state is not None:
        return int(random_state)
    if random_seed is not None:
        return int(random_seed)
    return None


def _derive_run_plan(
    total_steps: int,
    quick_mode: bool,
    exchange_override: int | None,
) -> tuple[int, int, int]:
    """Compute equilibration length, exchange frequency, and DCD stride.

    Exchange frequency optimization based on benchmarks:
    - Too frequent (10-50 steps): High overhead from exchange attempts
    - Optimal range (100-200 steps): Best balance of exchange rate and throughput
    - Too infrequent (500+ steps): Poor exchange statistics
    """

    total = int(total_steps)
    equilibration = min(total // 10, 200 if total <= 2000 else 2000)
    dcd_stride = max(1, total // 5000)
    exchange_frequency = max(100, total // 20)

    if quick_mode:
        equilibration = min(total // 20, 100)
        # Benchmark shows 100 steps is optimal, don't go lower even in quick mode
        exchange_frequency = max(100, total // 40)  # Changed from 50 to 100
        dcd_stride = max(1, total // 1000)

    if exchange_override is not None:
        override = int(exchange_override)
        if override > 0:
            exchange_frequency = override

    return equilibration, exchange_frequency, dcd_stride


def _emit_banner(
    title: str,
    lines: Iterable[str] | None = None,
    *,
    log_level: Literal["info", "warning"] = "info",
) -> None:
    """Emit a console/log banner with consistent formatting."""

    border = "=" * 80
    payload = list(lines or [])
    block = [border, title, border, *payload, border]
    print("\n" + "\n".join(block) + "\n", flush=True)
    log_fn = getattr(logger, log_level)
    for entry in block:
        log_fn(entry)


def _configure_cv_model(
    remd: ReplicaExchangeProtocol,
    cv_model_path: str | Path | None,
    cv_scaler_mean: Any | None,
    cv_scaler_scale: Any | None,
) -> None:
    """Attach optional CV model configuration to the REMD object."""

    if cv_model_path is None:
        return

    logger.info("[remd] Configuring CV model: %s", Path(cv_model_path).name)
    remd.cv_model_path = str(cv_model_path)

    if cv_scaler_mean is not None:
        remd.cv_scaler_mean = np.asarray(cv_scaler_mean, dtype=np.float64)
        logger.debug("[remd] CV scaler mean configured: shape=%s", remd.cv_scaler_mean.shape)

    if cv_scaler_scale is not None:
        remd.cv_scaler_scale = np.asarray(cv_scaler_scale, dtype=np.float64)
        logger.debug("[remd] CV scaler scale configured: shape=%s", remd.cv_scaler_scale.shape)


def _restore_from_checkpoint(
    remd: ReplicaExchangeProtocol,
    checkpoint_path: str | Path,
) -> bool:
    """Attempt to restore REMD state from ``checkpoint_path``."""

    logger.debug("[remd] Loading checkpoint from: %s", checkpoint_path)
    try:
        with open(checkpoint_path, "rb") as fh:
            ckpt = pickle.load(fh)
    except Exception as exc:
        logger.warning(
            "[remd] Failed to restore from checkpoint %s: %s",
            str(checkpoint_path),
            exc,
        )
        return False

    remd.restore_from_checkpoint(ckpt)
    logger.debug("[remd] Checkpoint restoration completed")
    return True


def _evaluate_demux_result(
    remd: ReplicaExchangeProtocol,
    demuxed_path: str | Path | None,
    total_steps: int,
    equilibration_steps: int,
    pdb_file: str | Path,
) -> tuple[bool, int | None]:
    """Return ``(accepted, frame_count)`` for a demultiplexed trajectory."""

    if not demuxed_path:
        logger.debug("[remd:demux] No demultiplexed trajectory path provided")
        return False, None

    logger.debug("[remd:demux] Evaluating trajectory: %s", Path(demuxed_path).name)

    try:
        reader = MDTrajReader(topology_path=str(pdb_file))
        nframes = reader.probe_length(str(demuxed_path))
        logger.debug("[remd:demux] Trajectory contains %d frames", nframes)

        reporter_stride = getattr(remd, "reporter_stride", None)
        effective_stride = int(
            reporter_stride
            if reporter_stride
            else max(1, getattr(remd, "dcd_stride", 1))
        )
        production_steps = max(0, int(total_steps) - int(equilibration_steps))
        expected_frames = max(1, production_steps // effective_stride)
        threshold = max(1, expected_frames // 5)

        logger.debug(
            "[remd:demux] Quality check: frames=%d, expected=%d, threshold=%d (20%% of expected)",
            nframes,
            expected_frames,
            threshold,
        )

        if int(nframes) >= threshold:
            logger.info(
                "[remd:demux] Trajectory accepted: %d frames (%.1f%% of expected)",
                nframes,
                100.0 * nframes / expected_frames if expected_frames > 0 else 0.0,
            )
            return True, int(nframes)
        else:
            logger.warning(
                "[remd:demux] Trajectory rejected: %d frames < %d threshold (%.1f%% of expected)",
                nframes,
                threshold,
                100.0 * nframes / expected_frames if expected_frames > 0 else 0.0,
            )
    except Exception as exc:  # pragma: no cover - diagnostic aid
        logger.debug("[remd:demux] Evaluation failed: %s", exc, exc_info=True)

    return False, None
