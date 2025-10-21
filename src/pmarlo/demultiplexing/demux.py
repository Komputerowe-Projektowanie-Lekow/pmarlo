"""
Demultiplexing utilities for Replica Exchange trajectories.

This module contains a standalone implementation of the demultiplexing
logic previously embedded in `replica_exchange.py` for better visibility
and future refactoring.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Literal, Mapping, Optional, cast

import numpy as np
from openmm import unit  # type: ignore

from pmarlo.io.trajectory_reader import TrajectoryIOError, get_reader
from pmarlo.io.trajectory_writer import get_writer
from pmarlo.transform.progress import ProgressCB
from pmarlo.utils.logging_utils import (
    announce_stage_complete,
    announce_stage_failed,
    emit_banner,
    format_duration,
)

from ..replica_exchange import config as _cfg
from .demux_engine import demux_streaming
from .demux_metadata import DemuxIntegrityError, DemuxMetadataDict, serialize_metadata
from .demux_plan import build_demux_plan
from .exchange_validation import normalize_exchange_mapping

logger = logging.getLogger("pmarlo")


FillPolicy = Literal["repeat", "skip", "interpolate"]


def _expected_frame_total(plan: Any) -> int:
    total = getattr(plan, "total_expected_frames", None)
    if total is not None:
        total_value = int(total)
        if total_value > 0:
            return total_value
    segments = getattr(plan, "segments")
    return int(sum(int(seg.expected_frames) for seg in segments))


def _select_target_temperature(
    remd: Any, target_temperature: float
) -> tuple[int, float]:
    temps = np.asarray(remd.temperatures, dtype=float)
    idx = int(np.argmin(np.abs(temps - float(target_temperature))))
    return idx, float(temps[idx])


def _determine_default_stride(remd: Any) -> int:
    if getattr(remd, "reporter_stride", None) is not None:
        return int(remd.reporter_stride)
    return int(max(1, getattr(remd, "dcd_stride", 1)))


def demux_trajectories(
    remd: Any,
    *,
    target_temperature: float = 300.0,
    equilibration_steps: int = 100,
    progress_callback: ProgressCB | None = None,
) -> Optional[str]:
    """Demultiplex trajectories to extract frames at a target temperature.

    This facade always routes to the streaming demux engine and raises if the
    streaming path encounters an unexpected error. The implementation no longer
    provides alternate code paths or silent degradations.

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
        If the exchange history maps to non-monotonic frame indices.
    """

    target_temp_idx, actual_temp = _select_target_temperature(remd, target_temperature)
    exchange_segments = len(remd.exchange_history or [])

    emit_banner(
        "PHASE 2/2: DEMULTIPLEXING STARTED (EXTRACTING FRAMES)",
        logger=logger,
        details=[
            f"Target temperature: {target_temperature:.1f} K",
            f"Closest available ladder entry: {actual_temp:.1f} K",
            f"Replica count: {remd.n_replicas}",
            f"Exchange segments available: {exchange_segments}",
        ],
    )

    if not remd.exchange_history:
        announce_stage_failed(
            "Demultiplexing",
            logger=logger,
            details=["No exchange history is available to demultiplex."],
        )
        logger.warning("No exchange history available for demultiplexing")
        return None

    default_stride = _determine_default_stride(remd)
    phase_start = perf_counter()
    try:
        result = _run_streaming_demux(
            remd,
            target_temperature,
            actual_temp,
            target_temp_idx,
            default_stride,
            equilibration_steps,
            progress_callback,
        )
    except DemuxIntegrityError as exc:
        announce_stage_failed(
            "Demultiplexing",
            logger=logger,
            details=[f"Integrity error encountered: {exc}"],
        )
        raise
    except Exception as exc:
        announce_stage_failed(
            "Demultiplexing",
            logger=logger,
            details=[f"Unexpected error during demultiplexing: {exc}"],
        )
        logger.exception("Streaming demux failed; aborting demultiplexing")
        raise

    elapsed = perf_counter() - phase_start
    if result:
        announce_stage_complete(
            "Demultiplexing",
            logger=logger,
            details=[
                f"Demultiplexed trajectory saved to: {result}",
                f"Duration: {format_duration(elapsed)}",
            ],
        )
    else:
        announce_stage_complete(
            "Demultiplexing",
            logger=logger,
            details=[
                "No frames were produced by the demultiplexing engine.",
                f"Duration: {format_duration(elapsed)}",
            ],
        )
    return result


def _run_streaming_demux(
    remd: Any,
    target_temperature: float,
    actual_temp: float,
    target_temp_idx: int,
    default_stride: int,
    equilibration_steps: int,
    progress_callback: ProgressCB | None,
) -> Optional[str]:
    temp_schedule = _build_temperature_schedule(remd)
    effective_equil_steps = _compute_streaming_equilibration_steps(equilibration_steps)
    _validate_exchange_integrity(
        remd,
        target_temp_idx,
        default_stride,
        effective_equil_steps,
    )

    backend = _resolve_backend(remd)
    reader = _configure_reader(backend, remd, warn_label="DEMUX chunk size")
    replica_paths, replica_frames, had_reader_error = _probe_replica_info(remd, reader)
    if not replica_paths:
        logger.warning("No replica trajectories provided; skipping demultiplexing")
        return None
    if not any(count > 0 for count in replica_frames):
        if had_reader_error:
            logger.warning(
                "No readable replica trajectories were found; skipping demultiplexing"
            )
        else:
            logger.warning(
                "Replica trajectories reported zero frames; skipping demultiplexing"
            )
        return None
    replica_strides = _resolve_replica_strides(remd)

    plan = build_demux_plan(
        exchange_history=remd.exchange_history,
        temperatures=remd.temperatures,
        target_temperature=float(target_temperature),
        exchange_frequency=int(remd.exchange_frequency),
        equilibration_offset=int(effective_equil_steps),
        replica_paths=replica_paths,
        replica_frames=replica_frames,
        default_stride=int(default_stride),
        replica_strides=replica_strides,
    )

    emit_banner(
        "PHASE 2/2: DEMULTIPLEXING - STREAMING FRAMES",
        logger=logger,
        details=[
            f"Total segments to process: {len(plan.segments)}",
            ("Expected output frames: ~" f"{_expected_frame_total(plan)}"),
        ],
    )

    demux_file = remd.output_dir / f"demux_T{actual_temp:.0f}K.dcd"

    # Console output
    print(f"Writing demultiplexed trajectory to: {demux_file.name}", flush=True)

    # Also log
    logger.info(f"Writing demultiplexed trajectory to: {demux_file.name}")

    writer = _open_demux_writer(remd, backend, demux_file)
    fill_policy = _resolve_fill_policy(remd)
    parallel_workers = _resolve_parallel_workers(remd)
    flush_between, checkpoint_every = _resolve_flush_settings(remd)

    stream_start = perf_counter()
    # Console output
    print("\n" + "=" * 80, flush=True)
    print("STREAMING FRAMES FROM REPLICA TRAJECTORIES...", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Also log
    logger.info("=" * 80)
    logger.info("STREAMING FRAMES FROM REPLICA TRAJECTORIES...")
    logger.info("=" * 80)

    result = demux_streaming(
        plan,
        str(remd.pdb_file),
        reader,
        writer,
        fill_policy=fill_policy,
        parallel_read_workers=parallel_workers,
        progress_callback=progress_callback,
        checkpoint_interval_segments=checkpoint_every,
        flush_between_segments=flush_between,
    )
    writer.close()
    stream_elapsed = perf_counter() - stream_start

    # Console output
    print("\n" + "=" * 80, flush=True)
    print("DEMULTIPLEXING COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"Total frames written: {result.total_frames_written}", flush=True)
    print(f"Output file: {demux_file}", flush=True)
    print(f"Elapsed: {format_duration(stream_elapsed)}", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Also log
    logger.info("=" * 80)
    logger.info("DEMULTIPLEXING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total frames written: {result.total_frames_written}")
    logger.info(f"Output file: {demux_file}")
    logger.info(f"Elapsed: {format_duration(stream_elapsed)}")
    logger.info("=" * 80)

    if int(result.total_frames_written) <= 0:
        logger.warning("Streaming demux produced 0 frames; no output written")
        return None

    timestep_ps = _integration_timestep_ps(remd)
    runtime_info, frames_mode = _build_runtime_info(
        remd,
        plan,
        fill_policy,
        effective_equil_steps,
        temp_schedule,
        timestep_ps,
    )
    meta_dict: DemuxMetadataDict = serialize_metadata(result, plan, runtime_info)
    meta_dict = _finalize_metadata_dict(
        meta_dict, plan, result, fill_policy, frames_mode
    )
    _write_metadata_file(demux_file, meta_dict, mode="streaming")
    if result.repaired_segments:
        logger.warning(f"Repaired segments: {result.repaired_segments}")
    return str(demux_file)


def _build_temperature_schedule(remd: Any) -> dict[str, dict[str, float]]:
    schedule: dict[str, dict[str, float]] = {
        str(i): {} for i in range(int(remd.n_replicas))
    }
    for step_index, states in enumerate(remd.exchange_history):
        for replica_idx, temp_idx in enumerate(states):
            schedule[str(replica_idx)][str(step_index)] = float(
                remd.temperatures[int(temp_idx)]
            )
    return schedule


def _compute_streaming_equilibration_steps(equilibration_steps: int) -> int:
    if equilibration_steps <= 0:
        return 0
    fast = max(100, equilibration_steps * 40 // 100)
    slow = max(100, equilibration_steps * 60 // 100)
    return int(fast + slow)


def _validate_exchange_integrity(
    remd: Any,
    target_temp_idx: int,
    default_stride: int,
    equilibration_steps: int,
) -> None:
    expected_prev_stop = 0
    for segment_index, states in enumerate(remd.exchange_history):
        normalized_states = normalize_exchange_mapping(
            states,
            expected_size=int(remd.n_replicas),
            context=f"segment {segment_index}",
            error_cls=DemuxIntegrityError,
        )

        replica_at_target = None
        for ridx, tidx in enumerate(normalized_states):
            if int(tidx) == int(target_temp_idx):
                replica_at_target = int(ridx)
                break
        if replica_at_target is None:
            continue

        stride_chk = _replica_stride(remd, replica_at_target, default_stride)
        try:
            if int(stride_chk) > int(remd.exchange_frequency):
                raise DemuxIntegrityError("Reporter stride exceeds exchange frequency")
        except DemuxIntegrityError:
            raise
        except (TypeError, ValueError):
            raise DemuxIntegrityError("Non-monotonic frame indices detected")

        start_md_chk = int(
            equilibration_steps + segment_index * remd.exchange_frequency
        )
        stop_md_chk = int(
            equilibration_steps + (segment_index + 1) * remd.exchange_frequency
        )
        start_frame_chk = max(
            0, (start_md_chk + int(stride_chk) - 1) // int(stride_chk)
        )
        end_frame_chk = max(0, (stop_md_chk + int(stride_chk) - 1) // int(stride_chk))
        if start_frame_chk < expected_prev_stop:
            raise DemuxIntegrityError("Non-monotonic frame indices detected")
        expected_prev_stop = max(expected_prev_stop, end_frame_chk)


def _replica_stride(remd: Any, replica_index: int, default_stride: int) -> int:
    strides = getattr(remd, "_replica_reporter_stride", []) or []
    if replica_index < len(strides):
        return int(strides[replica_index])
    return int(default_stride)


def _resolve_backend(remd: Any) -> str:
    backend = (
        getattr(remd, "demux_backend", None)
        or getattr(remd, "demux_io_backend", None)
        or getattr(_cfg, "DEMUX_BACKEND", getattr(_cfg, "DEMUX_IO_BACKEND", "mdtraj"))
    )
    return str(backend)


def _configure_reader(backend: str, remd: Any, *, warn_label: str) -> Any:
    reader = get_reader(str(backend), topology_path=str(remd.pdb_file))
    chunk_size = _resolve_buffer_setting(remd, warn_label)
    if chunk_size is not None and hasattr(reader, "chunk_size"):
        setattr(reader, "chunk_size", chunk_size)
    return reader


def _probe_replica_info(remd: Any, reader: Any) -> tuple[list[str], list[int], bool]:
    replica_paths: list[str] = []
    replica_frames: list[int] = []
    had_reader_error = False
    for path in remd.trajectory_files:
        str_path = str(path)
        replica_paths.append(str_path)
        try:
            frame_count = reader.probe_length(str_path)
        except FileNotFoundError:
            logger.warning(
                "Replica trajectory %s is missing; treating as zero available frames",
                str_path,
            )
            replica_frames.append(0)
            continue
        except TrajectoryIOError as exc:
            if _missing_file_error(exc):
                logger.warning(
                    "Replica trajectory %s is missing; treating as zero available frames",
                    str_path,
                )
                replica_frames.append(0)
                continue
            raise RuntimeError(
                f"Failed to probe frame count for replica trajectory {str_path}"
            ) from exc
        replica_frames.append(int(frame_count))
    return replica_paths, replica_frames, had_reader_error


def _missing_file_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, FileNotFoundError):
            return True
        cause = current.__cause__
        context = current.__context__ if cause is None else None
        current = cause if cause is not None else context
    return False


def _resolve_replica_strides(remd: Any) -> list[int] | None:
    strides = [int(s) for s in getattr(remd, "_replica_reporter_stride", [])]
    return strides if strides else None


def _open_demux_writer(remd: Any, backend: str, demux_file: Path) -> Any:
    writer = get_writer(str(backend), topology_path=str(remd.pdb_file))
    rewrite_threshold = _resolve_buffer_setting(remd, "DEMUX rewrite threshold")
    if rewrite_threshold is not None and hasattr(writer, "rewrite_threshold"):
        setattr(writer, "rewrite_threshold", rewrite_threshold)
    return writer.open(str(demux_file), str(remd.pdb_file), overwrite=True)


def _resolve_fill_policy(remd: Any) -> FillPolicy:
    raw = getattr(remd, "demux_fill_policy", None) or getattr(
        _cfg, "DEMUX_FILL_POLICY", "repeat"
    )
    if not isinstance(raw, str) or not raw:
        raise ValueError("demux_fill_policy must be a non-empty string")
    raw = raw.lower()
    if raw not in ("repeat", "skip", "interpolate"):
        raise ValueError(
            "demux_fill_policy must be one of {'repeat', 'skip', 'interpolate'}"
        )
    return cast(FillPolicy, raw)


def _resolve_parallel_workers(remd: Any) -> Optional[int]:
    workers = getattr(remd, "demux_parallel_workers", None)
    if workers is None:
        workers = getattr(_cfg, "DEMUX_PARALLEL_WORKERS", None)
    if workers is None:
        return None
    value = int(workers)
    if value <= 0:
        raise ValueError("demux_parallel_workers must be a positive integer")
    return value


def _resolve_flush_settings(remd: Any) -> tuple[bool, Optional[int]]:
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
    if checkpoint_every is None:
        return flush_between, None
    value = int(checkpoint_every)
    if value <= 0:
        raise ValueError("demux_checkpoint_interval must be a positive integer")
    return flush_between, value


def _integration_timestep_ps(remd: Any) -> float:
    integrators = getattr(remd, "integrators", None)
    if integrators:
        step = integrators[0].getStepSize()
        return float(step.value_in_unit(unit.picoseconds))
    return 0.0


def _build_runtime_info(
    remd: Any,
    plan: Any,
    fill_policy: FillPolicy,
    equilibration_steps: int,
    temperature_schedule: dict[str, dict[str, float]],
    timestep_ps: float,
) -> tuple[Dict[str, Any], int]:
    counts = Counter(int(seg.expected_frames) for seg in plan.segments)
    frames_mode = int(counts.most_common(1)[0][0]) if counts else 0
    runtime_info: Dict[str, Any] = {
        "exchange_frequency_steps": int(remd.exchange_frequency),
        "integration_timestep_ps": timestep_ps,
        "fill_policy": fill_policy,
        "temperature_schedule": temperature_schedule,
        "frames_per_segment": frames_mode,
        "equilibration_steps_total": int(equilibration_steps),
        "overlap_corrections": [],
    }
    return runtime_info, frames_mode


def _finalize_metadata_dict(
    meta_dict: Any,
    plan: Any,
    result: Any,
    fill_policy: FillPolicy,
    frames_mode: int,
) -> DemuxMetadataDict:
    if isinstance(meta_dict, dict):
        meta: Dict[str, Any] = dict(meta_dict)
    else:
        meta = {}
    meta.setdefault("schema_version", 2)
    meta.setdefault("segment_count", len(plan.segments))
    meta.setdefault("frames_per_segment", frames_mode)
    meta.setdefault("fill_policy", fill_policy)
    segments = list(plan.segments)
    real = list(result.segment_real_frames)
    if len(real) != len(segments):
        raise ValueError(
            "segment_real_frames length does not match demux plan segments"
        )
    repaired = set(result.repaired_segments)
    blocks: list[list[int]] = []
    collected = 0
    start: Optional[int] = None
    for index, segment in enumerate(segments):
        expected = int(segment.expected_frames)
        real_frames = int(real[index])
        is_repaired = (index in repaired) or (real_frames < expected)
        if expected <= 0 or is_repaired:
            if start is not None:
                blocks.append([int(start), int(collected)])
                start = None
        else:
            if start is None:
                start = collected
        collected += expected
    if start is not None:
        blocks.append([int(start), int(collected)])
    if blocks:
        meta.setdefault("contiguous_blocks", blocks)
    return cast(DemuxMetadataDict, meta)


def _write_metadata_file(
    demux_file: Path, metadata: Mapping[str, Any], *, mode: str
) -> None:
    meta_path = demux_file.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Demultiplexed ({mode}) saved: {demux_file}")
    logger.info(f"Metadata v2 saved: {meta_path}")


def _resolve_buffer_setting(remd: Any, warn_label: str) -> Optional[int]:
    raw = getattr(remd, "demux_chunk_size", None)
    if raw is None:
        raw = getattr(_cfg, "DEMUX_CHUNK_SIZE", None)
    if raw is None:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{warn_label} must be a positive integer")
    if value > 65536:
        logger.warning("%s too large (%d); clamping to 65536", warn_label, value)
        value = 65536
    return value
