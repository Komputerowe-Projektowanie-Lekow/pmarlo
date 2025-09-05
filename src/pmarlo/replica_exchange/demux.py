"""
Demultiplexing utilities for Replica Exchange trajectories.

This module contains a standalone implementation of the demultiplexing
logic previously embedded in `replica_exchange.py` for better visibility
and future refactoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from openmm import unit  # type: ignore

from pmarlo.progress import ProgressCB, ProgressReporter

from .demux_metadata import DemuxIntegrityError, DemuxMetadata


logger = logging.getLogger("pmarlo")


def demux_trajectories(
    remd: Any,
    *,
    target_temperature: float = 300.0,
    equilibration_steps: int = 100,
    progress_callback: ProgressCB | None = None,
) -> Optional[str]:
    """
    Demultiplex trajectories to extract frames at target temperature.

    Parameters
    ----------
    remd:
        An instance of `ReplicaExchange` holding simulation state and outputs.
    target_temperature:
        Target temperature to extract frames for.
    equilibration_steps:
        Number of equilibration steps (needed for frame calculation).
    progress_callback:
        Optional callback for progress reporting.

    Returns
    -------
    Optional[str]
        Path to the demultiplexed trajectory file, or None if failed.
    """

    reporter = ProgressReporter(progress_callback)
    logger.info(f"Demultiplexing trajectories for T = {target_temperature} K")

    # Find the target temperature index
    target_temp_idx = np.argmin(np.abs(np.array(remd.temperatures) - target_temperature))
    actual_temp = remd.temperatures[int(target_temp_idx)]

    logger.info(f"Using closest temperature: {actual_temp:.1f} K")

    # Check if we have exchange history
    if not remd.exchange_history:
        logger.warning("No exchange history available for demultiplexing")
        return None

    # Reporter stride: prefer per-replica recorded stride, otherwise use planned stride
    default_stride = int(remd.reporter_stride if remd.reporter_stride is not None else max(1, remd.dcd_stride))

    # Load all trajectories and perform segment-wise demultiplexing
    demux_segments: List[Any] = []
    trajectory_frame_counts: Dict[str, int] = {}
    repaired_segments: List[int] = []

    n_segments = len(remd.exchange_history)
    logger.info(f"Processing {n_segments} exchange steps (segments)...")
    # Provide progress-normalized fields so console can show ETA
    reporter.emit("demux_begin", {"segments": int(n_segments), "current": 0, "total": int(max(1, n_segments))})
    logger.info(
        (
            f"Exchange frequency: {remd.exchange_frequency} MD steps, "
            f"default DCD stride: {default_stride} MD steps"
        )
    )

    # Diagnostics for DCD files (silence plugin chatter while loading)
    logger.info("DCD File Diagnostics:")
    loaded_trajs: Dict[int, Any] = {}
    for i, traj_file in enumerate(remd.trajectory_files):
        if traj_file.exists():
            file_size = traj_file.stat().st_size
            logger.info(f"  Replica {i}: {traj_file.name} exists, size: {file_size:,} bytes")
            try:
                import mdtraj as md  # type: ignore

                from pmarlo.io.trajectory import _suppress_plugin_output  # type: ignore

                with _suppress_plugin_output():
                    t = md.load(str(traj_file), top=remd.pdb_file)
                loaded_trajs[i] = t
                trajectory_frame_counts[str(traj_file)] = int(t.n_frames)
                logger.info(f"    -> Loaded: {t.n_frames} frames")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"    -> Failed to load: {e}")
                trajectory_frame_counts[str(traj_file)] = 0
        else:
            logger.warning(f"  Replica {i}: {traj_file.name} does not exist")

    if not loaded_trajs:
        logger.warning("No trajectories could be loaded for demultiplexing")
        return None

    # Effective equilibration steps actually integrated (heating + temp equil)
    if equilibration_steps > 0:
        effective_equil_steps = max(100, equilibration_steps * 40 // 100) + max(100, equilibration_steps * 60 // 100)
    else:
        effective_equil_steps = 0

    # Prepare temperature schedule mapping
    temp_schedule: Dict[str, Dict[str, float]] = {str(rid): {} for rid in range(remd.n_replicas)}

    frames_per_segment: Optional[int] = None
    expected_start_frame = 0
    prev_stop_md = int(effective_equil_steps)

    # Build per-segment slices
    for s, replica_states in enumerate(remd.exchange_history):
        reporter.emit("demux_segment", {"index": int(s), "current": int(s + 1), "total": int(max(1, n_segments))})
        try:
            # Record temperature assignment for provenance
            for replica_idx, temp_state in enumerate(replica_states):
                temp_schedule[str(replica_idx)][str(s)] = float(remd.temperatures[int(temp_state)])

            # Which replica was at the target temperature during this segment
            replica_at_target: Optional[int] = None
            for replica_idx, temp_state in enumerate(replica_states):
                if int(temp_state) == int(target_temp_idx):
                    replica_at_target = int(replica_idx)
                    break

            # Segment MD step range [start, stop)
            start_md = int(effective_equil_steps + s * remd.exchange_frequency)
            stop_md = int(effective_equil_steps + (s + 1) * remd.exchange_frequency)

            if start_md < prev_stop_md:
                raise DemuxIntegrityError("Non-monotonic segment times detected")
            prev_stop_md = stop_md

            if replica_at_target is None:
                # Missing swap; fill using nearest neighbour if possible
                if demux_segments and frames_per_segment is not None:
                    import mdtraj as md  # type: ignore

                    fill = md.join([demux_segments[-1][-1:] for _ in range(int(frames_per_segment))])
                    demux_segments.append(fill)
                    repaired_segments.append(int(s))
                    expected_start_frame += int(frames_per_segment)
                    logger.warning(
                        f"Segment {s} missing target replica - filled with nearest neighbour frame"
                    )
                    continue
                raise DemuxIntegrityError(f"Segment {s} missing target replica and no data to fill")

            traj = loaded_trajs.get(replica_at_target)
            if traj is None:
                if demux_segments and frames_per_segment is not None:
                    import mdtraj as md  # type: ignore

                    fill = md.join([demux_segments[-1][-1:] for _ in range(int(frames_per_segment))])
                    demux_segments.append(fill)
                    repaired_segments.append(int(s))
                    expected_start_frame += int(frames_per_segment)
                    logger.warning(f"Segment {s} missing trajectory data - filled with nearest neighbour frame")
                    continue
                raise DemuxIntegrityError(f"Segment {s} missing trajectory data and no data to fill")

            # Map to saved frame indices using replica's recorded stride if available
            stride = (
                remd._replica_reporter_stride[replica_at_target]  # noqa: SLF001
                if replica_at_target < len(remd._replica_reporter_stride)  # noqa: SLF001
                else default_stride
            )
            start_frame = max(0, start_md // int(stride))
            # Inclusive of frames with step < stop_md
            end_frame = min(int(traj.n_frames), (max(0, stop_md - 1) // int(stride)) + 1)

            if start_frame > expected_start_frame:
                if demux_segments:
                    import mdtraj as md  # type: ignore

                    from pmarlo.io.trajectory import _suppress_plugin_output  # type: ignore

                    gap = start_frame - expected_start_frame
                    with _suppress_plugin_output():
                        fill = md.join([demux_segments[-1][-1:] for _ in range(int(gap))])
                    demux_segments.append(fill)
                    repaired_segments.append(int(s))
                    expected_start_frame = start_frame
                    logger.warning(f"Filled {gap} missing frame(s) before segment {s}")
                    reporter.emit(
                        "demux_gap_fill",
                        {"frames": int(gap), "current": int(min(s + 1, n_segments)), "total": int(max(1, n_segments))},
                    )
                else:
                    # Tolerate initial offset: start demux at the first available frame
                    gap = start_frame - expected_start_frame
                    expected_start_frame = start_frame
                    logger.warning(
                        f"Initial gap of {gap} frame(s) before first segment; starting at first available frame"
                    )
            elif start_frame < expected_start_frame:
                raise DemuxIntegrityError("Non-monotonic frame indices detected")

            if end_frame > start_frame:
                segment = traj[start_frame:end_frame]
                demux_segments.append(segment)
                if frames_per_segment is None:
                    frames_per_segment = int(end_frame - start_frame)
                expected_start_frame = end_frame
            else:
                if demux_segments and frames_per_segment is not None:
                    import mdtraj as md  # type: ignore

                    from pmarlo.io.trajectory import _suppress_plugin_output  # type: ignore

                    with _suppress_plugin_output():
                        fill = md.join([demux_segments[-1][-1:] for _ in range(int(frames_per_segment))])
                    demux_segments.append(fill)
                    repaired_segments.append(int(s))
                    expected_start_frame += int(frames_per_segment)
                    logger.warning(f"Segment {s} has no frames - filled with nearest neighbour frame")
                else:
                    raise DemuxIntegrityError(f"No frames available for segment {s}")
        except DemuxIntegrityError:
            raise
        except Exception as e:  # noqa: BLE001
            # Emit an explicit error event but continue demuxing subsequent segments
            reporter.emit(
                "demux_error",
                {
                    "index": int(s),
                    "error": str(e),
                    "current": int(min(s + 1, n_segments)),
                    "total": int(max(1, n_segments)),
                },
            )
            continue

    if demux_segments:
        try:
            import mdtraj as md  # type: ignore

            from pmarlo.io.trajectory import _suppress_plugin_output  # type: ignore

            with _suppress_plugin_output():
                demux_traj = md.join(demux_segments)
            demux_file = remd.output_dir / f"demux_T{actual_temp:.0f}K.dcd"
            with _suppress_plugin_output():
                demux_traj.save_dcd(str(demux_file))
            logger.info(f"Demultiplexed trajectory saved: {demux_file}")
            logger.info(f"Total frames at target temperature: {int(demux_traj.n_frames)}")
            reporter.emit(
                "demux_end",
                {"frames": int(demux_traj.n_frames), "file": str(demux_file), "current": int(n_segments), "total": int(max(1, n_segments))},
            )

            timestep_ps = float(
                remd.integrators[0].getStepSize().value_in_unit(unit.picoseconds) if remd.integrators else 0.0
            )
            metadata = DemuxMetadata(
                exchange_frequency_steps=int(remd.exchange_frequency),
                integration_timestep_ps=timestep_ps,
                frames_per_segment=int(frames_per_segment or 0),
                temperature_schedule=temp_schedule,
            )
            meta_path = demux_file.with_suffix(".meta.json")
            metadata.to_json(meta_path)
            logger.info(f"Demultiplexed metadata saved: {meta_path}")

            if repaired_segments:
                logger.warning(f"Repaired segments: {repaired_segments}")

            return str(demux_file)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error saving demultiplexed trajectory: {e}")
            return None
    else:
        logger.warning(
            (
                "No segments found for demultiplexing - check exchange history, "
                "frame indexing, or stride settings"
            )
        )
        logger.debug(f"  Exchange steps: {len(remd.exchange_history)}")
        logger.debug(f"  Exchange frequency: {remd.exchange_frequency}")
        logger.debug(f"  Effective equilibration steps: {effective_equil_steps}")
        logger.debug(f"  Default DCD stride: {default_stride}")
        for i, traj_file in enumerate(remd.trajectory_files):
            n_frames = trajectory_frame_counts.get(str(traj_file), 0)
            logger.debug(f"  Replica {i}: {n_frames} frames in {traj_file.name}")
        return None
