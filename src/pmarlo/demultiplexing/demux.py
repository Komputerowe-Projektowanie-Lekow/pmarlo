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
from typing import Any, Dict, Literal, Mapping, Optional, cast

import numpy as np
from openmm import unit  # type: ignore

from pmarlo.io.trajectory_reader import get_reader
from pmarlo.io.trajectory_writer import get_writer
from pmarlo.transform.progress import ProgressCB

from ..replica_exchange import config as _cfg
from .demux_engine import demux_streaming
from .demux_metadata import DemuxIntegrityError, DemuxMetadataDict, serialize_metadata
from .demux_plan import build_demux_plan
from .exchange_validation import normalize_exchange_mapping

logger = logging.getLogger("pmarlo")


FillPolicy = Literal["repeat", "skip", "interpolate"]


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


def _streaming_enabled() -> bool:
    try:
        return bool(getattr(_cfg, "DEMUX_STREAMING_ENABLED", True))
    except Exception:
        return True


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

    logger.info(f"Demultiplexing trajectories for T = {target_temperature} K")

    target_temp_idx, actual_temp = _select_target_temperature(remd, target_temperature)
    logger.info(f"Using closest temperature: {actual_temp:.1f} K")

    if not remd.exchange_history:
        logger.warning("No exchange history available for demultiplexing")
        return None

    default_stride = _determine_default_stride(remd)
    use_streaming = _streaming_enabled()

    if use_streaming:
        try:
            return _run_streaming_demux(
                remd,
                target_temperature,
                actual_temp,
                target_temp_idx,
                default_stride,
                equilibration_steps,
                progress_callback,
            )
        except DemuxIntegrityError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Streaming demux failed: {exc}")
            logger.info("Falling back to legacy demux implementation")

    try:
        return _run_legacy_demux(
            remd,
            target_temperature,
            actual_temp,
            default_stride,
            equilibration_steps,
            progress_callback,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Legacy demux failed: {exc}")
        return None


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
    replica_paths, replica_frames = _probe_replica_info(remd, reader)
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

    demux_file = remd.output_dir / f"demux_T{actual_temp:.0f}K.dcd"
    writer = _open_demux_writer(remd, backend, demux_file)
    fill_policy = _resolve_fill_policy(remd)
    parallel_workers = _resolve_parallel_workers(remd)
    flush_between, checkpoint_every = _resolve_flush_settings(remd)

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


def _run_legacy_demux(
    remd: Any,
    target_temperature: float,
    actual_temp: float,
    default_stride: int,
    equilibration_steps: int,
    progress_callback: ProgressCB | None,
) -> Optional[str]:
    temp_schedule = _build_temperature_schedule(remd)
    effective_equil_steps = int(equilibration_steps) if equilibration_steps else 0
    backend = _resolve_backend(remd)
    reader = _configure_reader(backend, remd, warn_label="DEMUX chunk size")
    replica_paths, replica_frames = _probe_replica_info(remd, reader)
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

    demux_file = remd.output_dir / f"demux_T{actual_temp:.0f}K.dcd"
    writer = _open_demux_writer(remd, backend, demux_file)
    fill_policy = _resolve_fill_policy(remd)
    parallel_workers = _resolve_parallel_workers(remd)
    flush_between, checkpoint_every = _resolve_flush_settings(remd)

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
    if int(result.total_frames_written) <= 0:
        logger.warning("Demux produced 0 frames in legacy path; no output written")
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
    _write_metadata_file(demux_file, meta_dict, mode="legacy")
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
        except Exception:
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
        try:
            return int(strides[replica_index])
        except Exception:
            return int(default_stride)
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


def _probe_replica_info(remd: Any, reader: Any) -> tuple[list[str], list[int]]:
    replica_paths: list[str] = []
    replica_frames: list[int] = []
    for path in remd.trajectory_files:
        replica_paths.append(str(path))
        try:
            frame_count = reader.probe_length(str(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not probe frames for {path}: {exc}")
            frame_count = 0
        replica_frames.append(int(frame_count))
    return replica_paths, replica_frames


def _resolve_replica_strides(remd: Any) -> list[int] | None:
    try:
        strides = [int(s) for s in getattr(remd, "_replica_reporter_stride", [])]
        return strides if strides else None
    except Exception:
        return None


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
        raw = "repeat"
    raw = raw.lower()
    if raw not in ("repeat", "skip", "interpolate"):
        raw = "repeat"
    return cast(FillPolicy, raw)


def _resolve_parallel_workers(remd: Any) -> Optional[int]:
    workers = getattr(remd, "demux_parallel_workers", None)
    if workers is None:
        workers = getattr(_cfg, "DEMUX_PARALLEL_WORKERS", None)
    try:
        if workers is None:
            return None
        value = int(workers)
        return value if value > 0 else None
    except Exception:
        return None


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
    try:
        if checkpoint_every is None:
            return flush_between, None
        value = int(checkpoint_every)
        return flush_between, value if value > 0 else None
    except Exception:
        return flush_between, None


def _integration_timestep_ps(remd: Any) -> float:
    try:
        if remd.integrators:
            return float(
                remd.integrators[0].getStepSize().value_in_unit(unit.picoseconds)
            )
    except Exception:
        pass
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
    meta.setdefault("segment_count", len(getattr(plan, "segments", []) or []))
    meta.setdefault("frames_per_segment", frames_mode)
    meta.setdefault("fill_policy", fill_policy)
    try:
        segments = list(getattr(plan, "segments", []) or [])
        real = list(getattr(result, "segment_real_frames", []) or [])
        repaired = set(getattr(result, "repaired_segments", []) or [])
        blocks: list[list[int]] = []
        collected = 0
        start: Optional[int] = None
        for index, segment in enumerate(segments):
            expected = int(getattr(segment, "expected_frames", 0) or 0)
            real_frames = int(real[index]) if index < len(real) else 0
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
    except Exception:
        pass
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
    try:
        if raw is None:
            return None
        value = int(raw)
    except Exception:
        return None
    if value <= 0:
        logger.warning("%s <= 0; coercing to 1", warn_label)
        value = 1
    if value > 65536:
        logger.warning("%s too large (%d); clamping to 65536", warn_label, value)
        value = 65536
    return value
