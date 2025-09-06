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

from . import config as _cfg
from .demux_metadata import DemuxIntegrityError, DemuxMetadata, serialize_metadata
from .demux_plan import build_demux_plan
from .demux_engine import demux_streaming
from pmarlo.io.trajectory_reader import get_reader
from pmarlo.io.trajectory_writer import get_writer


logger = logging.getLogger("pmarlo")


def demux_trajectories(
    remd: Any,
    *,
    target_temperature: float = 300.0,
    equilibration_steps: int = 100,
    progress_callback: ProgressCB | None = None,
) -> Optional[str]:
    """Demultiplex trajectories to extract frames at a target temperature.

    This facade routes to the streaming demux engine when the feature flag
    ``pmarlo.replica_exchange.config.DEMUX_STREAMING_ENABLED`` is True, and
    falls back to the legacy in‑memory implementation otherwise.

    Parameters
    ----------
    remd : Any
        An instance of ``ReplicaExchange`` holding simulation state and outputs.
    target_temperature : float, optional
        Target temperature in Kelvin to extract frames for (default 300.0).
    equilibration_steps : int, optional
        Number of equilibration steps used to compute the production offset
        in MD steps (default 100).
    progress_callback : ProgressCB or None, optional
        Optional callback for progress reporting. See ``pmarlo.progress``.

    Returns
    -------
    str or None
        Path to the demultiplexed trajectory file, or ``None`` if no frames
        could be produced.

    Raises
    ------
    DemuxIntegrityError
        If the exchange history maps to non‑monotonic frame indices.

    Examples
    --------
    Using a minimal REMD stub with prewritten DCDs (see tests for full flows)::

        remd = ReplicaExchange.__new__(ReplicaExchange)
        remd.pdb_file = "model.pdb"
        remd.temperatures = [300.0, 310.0]
        remd.n_replicas = 2
        remd.exchange_history = [[0, 1], [1, 0]]
        remd.reporter_stride = 1
        remd.dcd_stride = 1
        remd.exchange_frequency = 1
        remd.output_dir = Path("out")
        remd.integrators = []
        remd._replica_reporter_stride = [1, 1]
        remd.trajectory_files = [Path("replica_00.dcd"), Path("replica_01.dcd")]

        path = remd.demux_trajectories(target_temperature=300.0, equilibration_steps=0)
        # path is a string to the demuxed DCD or None
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

    # Optional streaming engine path
    try:
        use_streaming = bool(getattr(_cfg, "DEMUX_STREAMING_ENABLED", True))
    except Exception:
        use_streaming = True

    if use_streaming:
        try:
            # Build temperature schedule for metadata
            temp_schedule: Dict[str, Dict[str, float]] = {str(i): {} for i in range(int(remd.n_replicas))}
            for s, states in enumerate(remd.exchange_history):
                for ridx, tidx in enumerate(states):
                    temp_schedule[str(ridx)][str(s)] = float(remd.temperatures[int(tidx)])

            # Effective equilibration offset (MD steps)
            if equilibration_steps > 0:
                effective_equil_steps = max(100, equilibration_steps * 40 // 100) + max(100, equilibration_steps * 60 // 100)
            else:
                effective_equil_steps = 0

            # Preflight integrity check to preserve legacy semantics for
            # non-monotonic frame indices across segments
            expected_start_frame_check = 0
            for s, states in enumerate(remd.exchange_history):
                # locate replica at target temperature
                replica_at_target = None
                for ridx, tidx in enumerate(states):
                    if int(tidx) == int(target_temp_idx):
                        replica_at_target = int(ridx)
                        break
                if replica_at_target is None:
                    # Cannot verify this segment; continue
                    continue
                stride_chk = (
                    remd._replica_reporter_stride[replica_at_target]
                    if getattr(remd, "_replica_reporter_stride", []) and replica_at_target < len(remd._replica_reporter_stride)
                    else default_stride
                )
                start_md_chk = int((effective_equil_steps) + s * remd.exchange_frequency)
                stop_md_chk = int((effective_equil_steps) + (s + 1) * remd.exchange_frequency)
                start_frame_chk = max(0, start_md_chk // int(stride_chk))
                end_frame_chk = max(0, (max(0, stop_md_chk - 1) // int(stride_chk)) + 1)
                if start_frame_chk < expected_start_frame_check:
                    raise DemuxIntegrityError("Non-monotonic frame indices detected")
                if end_frame_chk > start_frame_chk:
                    expected_start_frame_check = end_frame_chk

            # Probe frame counts per replica
            # Resolve backend from instance overrides or config (supports both BACKEND and IO_BACKEND)
            backend = (
                getattr(remd, "demux_backend", None)
                or getattr(remd, "demux_io_backend", None)
                or getattr(_cfg, "DEMUX_BACKEND", getattr(_cfg, "DEMUX_IO_BACKEND", "mdtraj"))
            )
            reader = get_reader(str(backend), topology_path=str(remd.pdb_file))
            # Apply chunk size if supported by the reader
            try:
                chunk_size = getattr(remd, "demux_chunk_size", None)
                if chunk_size is None:
                    chunk_size = getattr(_cfg, "DEMUX_CHUNK_SIZE", None)
                if chunk_size is not None and hasattr(reader, "chunk_size"):
                    cs = int(chunk_size)
                    if cs <= 0:
                        logger.warning("DEMUX chunk size <= 0; coercing to 1")
                        cs = 1
                    if cs > 65536:
                        logger.warning("DEMUX chunk size too large (%d); clamping to 65536", cs)
                        cs = 65536
                    setattr(reader, "chunk_size", cs)
            except Exception:
                pass
            replica_frames: List[int] = []
            replica_paths: List[str] = []
            for p in remd.trajectory_files:
                replica_paths.append(str(p))
                try:
                    n = reader.probe_length(str(p))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Could not probe frames for {p}: {exc}")
                    n = 0
                replica_frames.append(int(n))

            # Per-replica stride list when available
            try:
                replica_strides = [int(s) for s in getattr(remd, "_replica_reporter_stride", [])]
            except Exception:
                replica_strides = []

            plan = build_demux_plan(
                exchange_history=remd.exchange_history,
                temperatures=remd.temperatures,
                target_temperature=float(target_temperature),
                exchange_frequency=int(remd.exchange_frequency),
                equilibration_offset=int(effective_equil_steps),
                replica_paths=replica_paths,
                replica_frames=replica_frames,
                default_stride=int(default_stride),
                replica_strides=replica_strides if replica_strides else None,
            )

            # Output path and writer
            demux_file = remd.output_dir / f"demux_T{actual_temp:.0f}K.dcd"
            writer = get_writer(str(backend), topology_path=str(remd.pdb_file))
            # Apply rewrite threshold if supported by the writer (mdtraj path)
            try:
                chunk_size = getattr(remd, "demux_chunk_size", None)
                if chunk_size is None:
                    chunk_size = getattr(_cfg, "DEMUX_CHUNK_SIZE", None)
                if chunk_size is not None and hasattr(writer, "rewrite_threshold"):
                    cs = int(chunk_size)
                    if cs <= 0:
                        logger.warning("DEMUX rewrite threshold <= 0; coercing to 1")
                        cs = 1
                    if cs > 65536:
                        logger.warning("DEMUX rewrite threshold too large (%d); clamping to 65536", cs)
                        cs = 65536
                    setattr(writer, "rewrite_threshold", cs)
            except Exception:
                pass
            writer = writer.open(
                str(demux_file), str(remd.pdb_file), overwrite=True
            )
            fill_policy = (getattr(remd, "demux_fill_policy", None) or getattr(_cfg, "DEMUX_FILL_POLICY", "repeat")).lower()
            # Resolve parallel workers
            parallel_workers = getattr(remd, "demux_parallel_workers", None)
            if parallel_workers is None:
                parallel_workers = getattr(_cfg, "DEMUX_PARALLEL_WORKERS", None)
            try:
                if parallel_workers is not None and int(parallel_workers) <= 0:
                    parallel_workers = None
            except Exception:
                parallel_workers = None
            # Resolve optional flush controls
            flush_between = bool(getattr(remd, "demux_flush_between_segments", getattr(_cfg, "DEMUX_FLUSH_BETWEEN_SEGMENTS", False)))
            checkpoint_every = getattr(remd, "demux_checkpoint_interval", getattr(_cfg, "DEMUX_CHECKPOINT_INTERVAL", None))
            try:
                checkpoint_every = int(checkpoint_every) if checkpoint_every is not None else None
                if checkpoint_every is not None and checkpoint_every <= 0:
                    checkpoint_every = None
            except Exception:
                checkpoint_every = None

            result = demux_streaming(
                plan,
                str(remd.pdb_file),
                reader,
                writer,
                fill_policy=fill_policy if fill_policy in {"repeat", "skip", "interpolate"} else "repeat",
                parallel_read_workers=parallel_workers,
                progress_callback=progress_callback,
                checkpoint_interval_segments=checkpoint_every,
                flush_between_segments=flush_between,
            )
            writer.close()
            if int(result.total_frames_written) <= 0:
                logger.warning("Streaming demux produced 0 frames; no output written")
                return None
            # Engine emitted demux_end; proceed to write metadata

            # Compute timestep and frames_per_segment override (mode)
            timestep_ps = float(
                remd.integrators[0].getStepSize().value_in_unit(unit.picoseconds) if remd.integrators else 0.0
            )
            # Mode of expected_frames across segments for better compatibility when variable
            from collections import Counter

            counts = Counter(int(seg.expected_frames) for seg in plan.segments)
            fps_mode = int(counts.most_common(1)[0][0]) if counts else 0
            runtime_info = {
                "exchange_frequency_steps": int(remd.exchange_frequency),
                "integration_timestep_ps": timestep_ps,
                "fill_policy": fill_policy,
                "temperature_schedule": temp_schedule,
                "frames_per_segment": fps_mode,
            }
            meta_dict = serialize_metadata(result, plan, runtime_info)
            # Safety: ensure required v2 keys are present
            if not isinstance(meta_dict, dict):
                meta_dict = {}
            meta_dict.setdefault("schema_version", 2)
            meta_dict.setdefault("segment_count", len(plan.segments))
            meta_dict.setdefault("frames_per_segment", fps_mode)
            meta_dict.setdefault("fill_policy", fill_policy)
            # Ensure contiguous_blocks present
            try:
                segs = getattr(plan, "segments", []) or []
                real = list(getattr(result, "segment_real_frames", []))
                repaired = set(getattr(result, "repaired_segments", []) or [])
                blocks = []
                pos = 0
                start = None
                for i, seg in enumerate(segs):
                    exp = int(getattr(seg, "expected_frames", 0) or 0)
                    r = int(real[i]) if i < len(real) else 0
                    is_rep = (i in repaired) or (r < exp)
                    if exp <= 0:
                        if start is not None:
                            blocks.append([int(start), int(pos)])
                            start = None
                    elif is_rep:
                        if start is not None:
                            blocks.append([int(start), int(pos)])
                            start = None
                    else:
                        if start is None:
                            start = pos
                    pos += exp
                if start is not None:
                    blocks.append([int(start), int(pos)])
                if blocks:
                    meta_dict.setdefault("contiguous_blocks", blocks)
            except Exception:
                pass
            meta_path = demux_file.with_suffix(".meta.json")
            meta_path.write_text(__import__("json").dumps(meta_dict, indent=2))
            logger.info(f"Demultiplexed (streaming) saved: {demux_file}")
            logger.info(f"Metadata v2 saved: {meta_path}")
            if result.repaired_segments:
                logger.warning(f"Repaired segments: {result.repaired_segments}")
            return str(demux_file)
        except DemuxIntegrityError:
            # Preserve legacy contract for integrity violations
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Streaming demux failed: {exc}")
            # Fall back to legacy implementation below
            logger.info("Falling back to legacy demux implementation")

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
            except (ImportError, OSError, RuntimeError, ValueError) as e:
                logger.warning(f"    -> Failed to load {traj_file} for replica {i}: {e}")
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
        except (ImportError, OSError, RuntimeError, ValueError) as e:
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
        except (OSError, RuntimeError, ValueError) as e:
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
