from typing import Any, Mapping, Tuple, Sequence, Optional, Dict, List, Callable
from pathlib import Path
import logging
import json
import hashlib

import numpy as np

from pmarlo.io.trajectory_writer import MDTrajDCDWriter
from pmarlo.io.trajectory_reader import MDTrajReader
from pmarlo.replica_exchange.demux_compat import (
    parse_exchange_log,
    parse_temperature_ladder,
)
from pmarlo.demultiplexing.exchange_validation import normalize_exchange_mapping
from pmarlo.utils.path_utils import ensure_directory

logger = logging.getLogger("pmarlo")


def demultiplex_run(
    run_id: str,
    replica_traj_paths: list[str | Path],
    exchange_log_path: str | Path,
    topology_path: str | Path,
    ladder_K: list[float] | str,
    dt_ps: float,
    out_dir: str | Path,
    fmt: str = "dcd",
    chunk_size: int = 5000,
) -> list[str]:
    """Demultiplex a REMD run into per-temperature trajectories and manifests.

    .. deprecated:: 0.0.42
        This function is deprecated. Use :func:`pmarlo.demultiplexing.demux.demux_trajectories`
        or the streaming demux functions directly.

    Returns list of DemuxShard JSON paths.
    """
    logger.info(
        "[demux] Starting demultiplexing: run_id=%s, n_replicas=%d, exchange_log=%s, format=%s",
        run_id,
        len(replica_traj_paths),
        exchange_log_path,
        fmt,
    )

    out_dir_path, topo_path, replica_paths = _prepare_demux_paths(
        out_dir,
        topology_path,
        replica_traj_paths,
    )
    temperatures = _parse_temperature_ladder_safe(ladder_K, parse_temperature_ladder)
    logger.info("[demux] Temperature ladder: %s", temperatures)

    exchange_records = _load_exchange_records_safe(
        exchange_log_path, parse_exchange_log
    )
    logger.info("[demux] Loaded %d exchange records", len(exchange_records))

    if not exchange_records:
        logger.warning("[demux] No exchange records found, returning empty result")
        return []

    _validate_demux_inputs(temperatures, replica_paths, exchange_records)
    logger.info("[demux] Input validation passed")

    logger.info("[demux] Loading replica trajectory frames...")
    reader = MDTrajReader(topology_path=str(topo_path))
    replica_frames = _collect_replica_frames(reader, replica_paths)
    total_frames = sum(len(frames) for frames in replica_frames)
    logger.info("[demux] Loaded %d total frames across %d replicas", total_frames, len(replica_frames))

    writers, dcd_paths = _open_demux_writers(
        out_dir_path,
        topo_path,
        temperatures,
        fmt,
    )
    logger.info("[demux] Opened %d output writers", len(writers))

    try:
        logger.info("[demux] Processing exchange segments...")
        segments_per_temp, dst_positions = _demux_exchange_segments(
            exchange_records,
            replica_frames,
            writers,
        )
        logger.info("[demux] Successfully processed all segments")
    finally:
        _close_demux_writers(writers)
        logger.debug("[demux] Closed all trajectory writers")

    logger.info("[demux] Writing manifests...")
    manifest_paths = _write_demux_manifests(
        run_id,
        temperatures,
        dcd_paths,
        dst_positions,
        segments_per_temp,
        dt_ps,
        topology_path=topo_path,
    )
    logger.info("[demux] Demultiplexing complete: generated %d manifests", len(manifest_paths))

    return manifest_paths


def _prepare_demux_paths(
    out_dir: str | Path,
    topology_path: str | Path,
    replica_traj_paths: list[str | Path],
) -> tuple[Path, Path, list[Path]]:
    """Create output directory and normalise key input paths."""
    logger.debug("[demux] Preparing paths: output_dir=%s, topology=%s", out_dir, topology_path)

    out_dir_path = Path(out_dir)
    ensure_directory(out_dir_path)
    topo_path = Path(topology_path)
    replica_paths = [Path(p) for p in replica_traj_paths]

    logger.debug("[demux] Created output directory: %s", out_dir_path)
    return out_dir_path, topo_path, replica_paths


def _parse_temperature_ladder_safe(
    ladder: list[float] | str,
    parser: Callable[[list[float] | str], Sequence[float]],
) -> list[float]:
    """Parse the temperature ladder and surface friendlier errors."""
    logger.debug("[demux] Parsing temperature ladder")

    values = list(parser(ladder))

    logger.debug("[demux] Successfully parsed %d temperatures", len(values))
    return [float(val) for val in values]


def _load_exchange_records_safe(
    exchange_log_path: str | Path,
    loader: Callable[[str], Sequence[Any]],
) -> list[Any]:
    """Load exchange records with consistent error handling."""
    logger.debug("[demux] Loading exchange log from: %s", exchange_log_path)

    records = list(loader(str(exchange_log_path)))

    records.sort(key=lambda rec: rec.step_index)
    logger.debug("[demux] Loaded and sorted %d exchange records", len(records))
    return records


def _validate_demux_inputs(
    temperatures: Sequence[float],
    replica_paths: Sequence[Path],
    exchange_records: Sequence[Any],
) -> None:
    """Sanity-check parsed inputs before demultiplexing frames."""

    if len(temperatures) != len(replica_paths):
        raise ValueError(
            "Temperature ladder length does not match number of replica trajectories"
        )
    if not exchange_records:
        raise ValueError("Exchange log contained no exchanges")
    n_temps = len(temperatures)
    if any(len(record.temp_to_replica) != n_temps for record in exchange_records):
        raise ValueError("Exchange log column count does not match temperature ladder")


def _collect_replica_frames(
    reader: Any,
    replica_paths: Sequence[Path],
) -> list[list[np.ndarray]]:
    """Load all frames for each replica using the shared reader."""
    logger.debug("[demux] Loading frames from %d replica trajectories", len(replica_paths))

    frames_per_replica: list[list[np.ndarray]] = []
    for idx, path in enumerate(replica_paths):
        count = reader.probe_length(str(path))
        logger.debug("[demux] Replica %d (%s): loading %d frames", idx, path.name, count)
        frames = list(reader.iter_frames(str(path), start=0, stop=count, stride=1))
        frames_per_replica.append(frames)

    logger.debug("[demux] Completed loading all replica frames")
    return frames_per_replica


def _open_demux_writers(
    out_dir_path: Path,
    topo_path: Path,
    temperatures: Sequence[float],
    fmt: str,
) -> tuple[list[MDTrajDCDWriter], list[Path]]:
    """Open one trajectory writer per temperature and return their paths."""
    logger.debug("[demux] Opening %d trajectory writers for format: %s", len(temperatures), fmt)

    writers: list[MDTrajDCDWriter] = []
    paths: list[Path] = []
    for temp in temperatures:
        demux_path = out_dir_path / f"demux_T{float(temp):.0f}K.{fmt}"
        logger.debug("[demux] Creating writer for T=%.0fK at %s", temp, demux_path)
        writer = MDTrajDCDWriter()
        writer.open(str(demux_path), topology_path=str(topo_path), overwrite=True)
        writers.append(writer)
        paths.append(demux_path)

    logger.debug("[demux] All writers opened successfully")
    return writers, paths


def _demux_exchange_segments(
    exchange_records: Sequence[Any],
    replica_frames: Sequence[Sequence[np.ndarray]],
    writers: Sequence[MDTrajDCDWriter],
) -> tuple[list[list[Dict[str, Any]]], list[int]]:
    """Replay exchanges and write per-temperature segments."""

    n_temps = len(writers)
    segments_per_temp: list[list[Dict[str, Any]]] = [list() for _ in range(n_temps)]
    dst_positions = [0] * n_temps
    segments_consumed = [0] * len(replica_frames)

    for seg_index, record in enumerate(exchange_records):
        mapping = normalize_exchange_mapping(
            record.temp_to_replica,
            expected_size=len(replica_frames),
            context=f"segment {seg_index}",
        )
        frame_index = seg_index // max(1, len(replica_frames))

        for temp_index, rep_idx in enumerate(mapping):
            frames_for_replica = replica_frames[rep_idx]
            if frame_index >= len(frames_for_replica):
                raise ValueError(
                    f"Replica {rep_idx} exhausted after {frame_index} frames"
                )

            segments_consumed[rep_idx] += 1
            if segments_consumed[rep_idx] > len(frames_for_replica):
                raise ValueError(
                    f"Replica {rep_idx} consumed more segments than available frames"
                )

            frame = frames_for_replica[frame_index]
            writers[temp_index].write_frames(frame[np.newaxis, :, :])

            src_start = frame_index
            dst_start = dst_positions[temp_index]
            segment_info = {
                "segment_index": int(seg_index),
                "slice_index": int(record.step_index),
                "source_replica": int(rep_idx),
                "src_frame_start": int(src_start),
                "src_frame_stop": int(src_start + 1),
                "dst_frame_start": int(dst_start),
                "dst_frame_stop": int(dst_start + 1),
            }
            segments_per_temp[temp_index].append(segment_info)
            dst_positions[temp_index] += 1

    return segments_per_temp, dst_positions


def _close_demux_writers(writers: Sequence[MDTrajDCDWriter]) -> None:
    """Close all trajectory writers."""
    logger.debug("[demux] Closing %d trajectory writers", len(writers))

    for writer in writers:
        writer.close()


def _write_demux_manifests(
    run_id: str,
    temperatures: Sequence[float],
    dcd_paths: Sequence[Path],
    dst_positions: Sequence[int],
    segments_per_temp: Sequence[Sequence[Dict[str, Any]]],
    dt_ps: float,
    topology_path: Path | None = None,
) -> list[str]:
    """Write JSON manifests for each demultiplexed temperature trajectory."""

    def _compute_digest(traj_path: Path) -> str:
        """Compute SHA256 digest of trajectory frames."""
        if topology_path is None:
            return hashlib.sha256(traj_path.read_bytes()).hexdigest()

        reader = MDTrajReader(topology_path=str(topology_path))
        total = reader.probe_length(str(traj_path))
        if total <= 0:
            return hashlib.sha256(b"").hexdigest()

        digest = hashlib.sha256()
        for frame in reader.iter_frames(
            str(traj_path), start=0, stop=total, stride=1
        ):
            data = np.ascontiguousarray(np.asarray(frame, dtype=np.float32))
            digest.update(data.tobytes())
        return digest.hexdigest()

    logger.debug("[demux] Writing %d manifest files", len(temperatures))

    json_paths: list[str] = []
    run_id_str = str(run_id)
    dt_ps_value = float(dt_ps)

    for temp_index, temp in enumerate(temperatures):
        dcd_path = dcd_paths[temp_index]
        logger.debug("[demux] Computing digest for T=%.0fK trajectory", temp)
        digest = _compute_digest(dcd_path)

        metadata = {
            "schema_version": "2.0",
            "kind": "demux",
            "run_id": run_id_str,
            "temperature_K": float(temp),
            "n_frames": int(dst_positions[temp_index]),
            "dt_ps": dt_ps_value,
            "trajectory": dcd_path.name,
            "segments": list(segments_per_temp[temp_index]),
            "integrity": {"traj_sha256": digest},
        }

        json_path = dcd_path.with_suffix(".json")
        json_path.write_text(json.dumps(metadata, sort_keys=True))
        json_paths.append(str(json_path))
        logger.debug("[demux] Wrote manifest: %s", json_path.name)

    return json_paths