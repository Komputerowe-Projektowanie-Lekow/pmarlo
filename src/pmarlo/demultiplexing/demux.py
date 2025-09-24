"""
Demultiplexing utilities for Replica Exchange trajectories.

This module contains a standalone implementation of the demultiplexing
logic previously embedded in `replica_exchange.py` for better visibility
and future refactoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
from openmm import unit  # type: ignore

from pmarlo.io.trajectory_reader import get_reader
from pmarlo.io.trajectory_writer import get_writer
from pmarlo.transform.progress import ProgressCB, ProgressReporter

from ..replica_exchange import config as _cfg
from .demux_engine import demux_streaming
from .demux_metadata import DemuxIntegrityError, serialize_metadata
from .demux_plan import build_demux_plan

logger = logging.getLogger("pmarlo")


FillPolicy = Literal["repeat", "skip", "interpolate"]


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
    """

    reporter = ProgressReporter(progress_callback)
    logger.info(f"Demultiplexing trajectories for T = {target_temperature} K")

    # Find the target temperature index
    target_temp_idx = np.argmin(
        np.abs(np.array(remd.temperatures) - target_temperature)
    )
    actual_temp = remd.temperatures[int(target_temp_idx)]

    logger.info(f"Using closest temperature: {actual_temp:.1f} K")

    # Check if we have exchange history
    if not remd.exchange_history:
        logger.warning("No exchange history available for demultiplexing")
        return None

    # Reporter stride: prefer per-replica recorded stride, otherwise use planned stride
    default_stride = int(
        remd.reporter_stride
        if remd.reporter_stride is not None
        else max(1, remd.dcd_stride)
    )

    # Optional streaming engine path
    try:
        use_streaming = bool(getattr(_cfg, "DEMUX_STREAMING_ENABLED", True))
    except Exception:
        use_streaming = True

    if use_streaming:
        try:
            # Build temperature schedule for metadata
            temp_schedule: dict[str, dict[str, float]] = {
                str(i): {} for i in range(int(remd.n_replicas))
            }
            for s, states in enumerate(remd.exchange_history):
                for ridx, tidx in enumerate(states):
                    temp_schedule[str(ridx)][str(s)] = float(
                        remd.temperatures[int(tidx)]
                    )

            # Effective equilibration offset (MD steps)
            if equilibration_steps > 0:
                effective_equil_steps = max(100, equilibration_steps * 40 // 100) + max(
                    100, equilibration_steps * 60 // 100
                )
            else:
                effective_equil_steps = 0

            # Preflight integrity check: detect obviously inconsistent metadata
            # Use ceil mapping (half-open windows) to avoid rounding backtracks.
            expected_prev_stop = 0
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
                    if getattr(remd, "_replica_reporter_stride", [])
                    and replica_at_target < len(remd._replica_reporter_stride)
                    else default_stride
                )
                # If stride exceeds exchange frequency, segment boundaries become ambiguous
                try:
                    if int(stride_chk) > int(remd.exchange_frequency):
                        raise DemuxIntegrityError(
                            "Reporter stride exceeds exchange frequency"
                        )
                except DemuxIntegrityError:
                    raise
                except Exception:
                    # Fall back to legacy message when types are odd
                    raise DemuxIntegrityError("Non-monotonic frame indices detected")

                start_md_chk = int(effective_equil_steps + s * remd.exchange_frequency)
                stop_md_chk = int(
                    effective_equil_steps + (s + 1) * remd.exchange_frequency
                )
                # ceil mapping for both boundaries
                start_frame_chk = max(
                    0, (start_md_chk + int(stride_chk) - 1) // int(stride_chk)
                )
                end_frame_chk = max(
                    0, (stop_md_chk + int(stride_chk) - 1) // int(stride_chk)
                )
                if start_frame_chk < expected_prev_stop:
                    # Should not happen with ceil mapping unless metadata is inconsistent
                    raise DemuxIntegrityError("Non-monotonic frame indices detected")
                expected_prev_stop = max(expected_prev_stop, end_frame_chk)

            # Probe frame counts per replica
            # Resolve backend from instance overrides or config (supports both BACKEND and IO_BACKEND)
            backend = (
                getattr(remd, "demux_backend", None)
                or getattr(remd, "demux_io_backend", None)
                or getattr(
                    _cfg, "DEMUX_BACKEND", getattr(_cfg, "DEMUX_IO_BACKEND", "mdtraj")
                )
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
                        logger.warning(
                            "DEMUX chunk size too large (%d); clamping to 65536", cs
                        )
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
                replica_strides = [
                    int(s) for s in getattr(remd, "_replica_reporter_stride", [])
                ]
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
                        logger.warning(
                            "DEMUX rewrite threshold too large (%d); clamping to 65536",
                            cs,
                        )
                        cs = 65536
                    setattr(writer, "rewrite_threshold", cs)
            except Exception:
                pass
            writer = writer.open(str(demux_file), str(remd.pdb_file), overwrite=True)
            raw_fill_policy = getattr(remd, "demux_fill_policy", None) or getattr(
                _cfg, "DEMUX_FILL_POLICY", "repeat"
            )
            if not isinstance(raw_fill_policy, str) or not raw_fill_policy:
                raw_fill_policy = "repeat"
            _raw = raw_fill_policy.lower()
            if _raw not in ("repeat", "skip", "interpolate"):
                _raw = "repeat"
            fill_policy_lit: FillPolicy = cast(FillPolicy, _raw)
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
            flush_between = bool(
                getattr(
                    remd,
                    "demux_flush_between_segments",
                    getattr(_cfg, "DEMUX_FLUSH_BETWEEN_SEGMENTS", False),
                )
            )
            checkpoint_every = getattr(
                remd,
                "demux_checkpoint_interval",
                getattr(_cfg, "DEMUX_CHECKPOINT_INTERVAL", None),
            )
            try:
                checkpoint_every = (
                    int(checkpoint_every) if checkpoint_every is not None else None
                )
                if checkpoint_every is not None and checkpoint_every <= 0:
                    checkpoint_every = None
            except Exception:
                checkpoint_every = None

            result = demux_streaming(
                plan,
                str(remd.pdb_file),
                reader,
                writer,
                fill_policy=fill_policy_lit,
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
                remd.integrators[0].getStepSize().value_in_unit(unit.picoseconds)
                if remd.integrators
                else 0.0
            )
            # Mode of expected_frames across segments for better compatibility when variable
            from collections import Counter

            counts = Counter(int(seg.expected_frames) for seg in plan.segments)
            fps_mode = int(counts.most_common(1)[0][0]) if counts else 0
            runtime_info_streaming: Dict[str, Any] = {
                "exchange_frequency_steps": int(remd.exchange_frequency),
                "integration_timestep_ps": timestep_ps,
                "fill_policy": fill_policy_lit,
                "temperature_schedule": temp_schedule,
                "frames_per_segment": fps_mode,
                "equilibration_steps_total": int(effective_equil_steps),
                "overlap_corrections": [],
            }
            meta_dict = serialize_metadata(result, plan, runtime_info_streaming)
            # Safety: ensure required v2 keys are present
            if not isinstance(meta_dict, dict):
                meta_dict = {}
            meta_dict.setdefault("schema_version", 2)
            meta_dict.setdefault("segment_count", len(plan.segments))
            meta_dict.setdefault("frames_per_segment", fps_mode)
            meta_dict.setdefault("fill_policy", fill_policy_lit)
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

    # Legacy fallback: build the same plan and use the same engine to ensure a path is returned
    try:
        # Build temperature schedule for metadata
        temp_schedule = {str(i): {} for i in range(int(remd.n_replicas))}
        for s, states in enumerate(remd.exchange_history):
            for ridx, tidx in enumerate(states):
                temp_schedule[str(ridx)][str(s)] = float(remd.temperatures[int(tidx)])

        effective_equil_steps = int(equilibration_steps) if equilibration_steps else 0

        # Probe frame counts per replica
        backend = (
            getattr(remd, "demux_backend", None)
            or getattr(remd, "demux_io_backend", None)
            or getattr(
                _cfg, "DEMUX_BACKEND", getattr(_cfg, "DEMUX_IO_BACKEND", "mdtraj")
            )
        )
        reader = get_reader(str(backend), topology_path=str(remd.pdb_file))
        try:
            chunk_size = getattr(remd, "demux_chunk_size", None)
            if chunk_size is None:
                chunk_size = getattr(_cfg, "DEMUX_CHUNK_SIZE", None)
            if chunk_size is not None and hasattr(reader, "chunk_size"):
                cs = int(chunk_size)
                if cs <= 0:
                    cs = 1
                if cs > 65536:
                    cs = 65536
                setattr(reader, "chunk_size", cs)
        except Exception:
            pass
        replica_frames = []
        replica_paths = []
        for p in remd.trajectory_files:
            replica_paths.append(str(p))
            try:
                n = reader.probe_length(str(p))
            except Exception:
                n = 0
            replica_frames.append(int(n))

        try:
            replica_strides = [
                int(s) for s in getattr(remd, "_replica_reporter_stride", [])
            ]
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

        demux_file = remd.output_dir / f"demux_T{actual_temp:.0f}K.dcd"
        writer = get_writer(str(backend), topology_path=str(remd.pdb_file))
        try:
            chunk_size = getattr(remd, "demux_chunk_size", None)
            if chunk_size is None:
                chunk_size = getattr(_cfg, "DEMUX_CHUNK_SIZE", None)
            if chunk_size is not None and hasattr(writer, "rewrite_threshold"):
                cs = int(chunk_size)
                if cs <= 0:
                    cs = 1
                if cs > 65536:
                    cs = 65536
                setattr(writer, "rewrite_threshold", cs)
        except Exception:
            pass
        writer = writer.open(str(demux_file), str(remd.pdb_file), overwrite=True)
        raw_fill_policy = getattr(remd, "demux_fill_policy", None) or getattr(
            _cfg, "DEMUX_FILL_POLICY", "repeat"
        )
        if not isinstance(raw_fill_policy, str) or not raw_fill_policy:
            raw_fill_policy = "repeat"
        _raw = raw_fill_policy.lower()
        if _raw not in ("repeat", "skip", "interpolate"):
            _raw = "repeat"
        legacy_fill_policy: FillPolicy = cast(FillPolicy, _raw)

        # Resolve parallel workers even for legacy path for parity; often unused
        parallel_workers = getattr(remd, "demux_parallel_workers", None)
        if parallel_workers is None:
            parallel_workers = getattr(_cfg, "DEMUX_PARALLEL_WORKERS", None)
        try:
            if parallel_workers is not None and int(parallel_workers) <= 0:
                parallel_workers = None
        except Exception:
            parallel_workers = None

        flush_between = bool(
            getattr(
                remd,
                "demux_flush_between_segments",
                getattr(_cfg, "DEMUX_FLUSH_BETWEEN_SEGMENTS", False),
            )
        )
        checkpoint_every = getattr(
            remd,
            "demux_checkpoint_interval",
            getattr(_cfg, "DEMUX_CHECKPOINT_INTERVAL", None),
        )
        try:
            checkpoint_every = (
                int(checkpoint_every) if checkpoint_every is not None else None
            )
            if checkpoint_every is not None and checkpoint_every <= 0:
                checkpoint_every = None
        except Exception:
            checkpoint_every = None

        result = demux_streaming(
            plan,
            str(remd.pdb_file),
            reader,
            writer,
            fill_policy=legacy_fill_policy,
            parallel_read_workers=parallel_workers,
            progress_callback=progress_callback,
            checkpoint_interval_segments=checkpoint_every,
            flush_between_segments=flush_between,
        )
        writer.close()
        if int(result.total_frames_written) <= 0:
            logger.warning("Demux produced 0 frames in legacy path; no output written")
            return None

        timestep_ps = float(
            remd.integrators[0].getStepSize().value_in_unit(unit.picoseconds)
            if remd.integrators
            else 0.0
        )
        from collections import Counter

        counts = Counter(int(seg.expected_frames) for seg in plan.segments)
        fps_mode = int(counts.most_common(1)[0][0]) if counts else 0
        runtime_info_legacy: Dict[str, Any] = {
            "exchange_frequency_steps": int(remd.exchange_frequency),
            "integration_timestep_ps": timestep_ps,
            "fill_policy": legacy_fill_policy,
            "temperature_schedule": temp_schedule,
            "frames_per_segment": fps_mode,
            "equilibration_steps_total": int(effective_equil_steps),
            "overlap_corrections": [],
        }
        meta_dict = serialize_metadata(result, plan, runtime_info_legacy)
        if not isinstance(meta_dict, dict):
            meta_dict = {}
        meta_dict.setdefault("schema_version", 2)
        meta_dict.setdefault("segment_count", len(plan.segments))
        meta_dict.setdefault("frames_per_segment", fps_mode)
        meta_dict.setdefault("fill_policy", legacy_fill_policy)
        # contiguous blocks
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
        logger.info(f"Demultiplexed (legacy path) saved: {demux_file}")
        logger.info(f"Metadata v2 saved: {meta_path}")
        if result.repaired_segments:
            logger.warning(f"Repaired segments: {result.repaired_segments}")
        return str(demux_file)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Legacy demux failed: {exc}")
        return None
