from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence, cast

import mdtraj as md

from pmarlo.utils.mdtraj import load_mdtraj_topology, resolve_atom_selection
from pmarlo.utils.path_utils import resolve_project_path


class LoadingMixin:
    # Attributes provided by host class
    trajectory_files: list[str]
    trajectories: list[md.Trajectory]
    topology_file: str | None
    demux_metadata: object | None
    frame_stride: int | None
    time_per_frame_ps: float | None
    _update_total_frames: Callable[[], None]

    def load_trajectories(
        self,
        *,
        stride: int = 1,
        atom_selection: str | Sequence[int] | None = None,
        chunk_size: int = 1000,
    ) -> None:
        """Load trajectory data for analysis in streaming mode.

        Trajectories are streamed from disk using mdtraj.iterload to avoid
        loading entire files into memory. Supports optional atom selection.
        """

        logger = getattr(self, "logger", None)
        if logger is None:
            import logging as _logging

            logger = _logging.getLogger("pmarlo")

        logger.info("Loading trajectory data (streaming mode)...")

        atom_indices = self._resolve_atom_indices(atom_selection)

        self.trajectories = []
        ignore_errors = getattr(self, "ignore_trajectory_errors", False)

        # Initialize counters for tracking skipped shards
        skipped_shard_count = 0
        empty_shard_count = 0
        first_skipped_shards = []
        first_empty_shards = []
        total_shards_processed = len(self.trajectory_files)
        MAX_DETAILED_LOGS = 5

        for i, traj_file in enumerate(self.trajectory_files):
            joined = self._stream_single_trajectory(
                traj_file=traj_file,
                stride=stride,
                atom_indices=atom_indices,
                chunk_size=chunk_size,
                selection_str=(
                    atom_selection if isinstance(atom_selection, str) else None
                ),
            )
            if joined is None:
                skipped_shard_count += 1
                # Only log details for first few skipped shards
                if skipped_shard_count <= MAX_DETAILED_LOGS:
                    logger.info(
                        "Trajectory %d (%s): SKIPPED - Loading failed or returned None",
                        i + 1,
                        traj_file,
                    )
                    # TEMPORARY DEBUG CHECK: Verify logger configuration
                    print(
                        f"DEBUG_CHECK: Logger name='{logger.name}', Effective level={logger.getEffectiveLevel()}, Handler count={len(logger.handlers)}, Root handler count={len(__import__('logging').getLogger().handlers)}"
                    )
                    first_skipped_shards.append(f"{i + 1} ({traj_file})")
                continue

            # Check if trajectory is empty (0 frames)
            if joined.n_frames == 0:
                empty_shard_count += 1
                # Only log details for first few empty shards
                if empty_shard_count <= MAX_DETAILED_LOGS:
                    logger.info(
                        "Trajectory %d (%s): SKIPPED - Empty trajectory (0 frames)",
                        i + 1,
                        traj_file,
                    )
                    # TEMPORARY DEBUG CHECK: Verify logger configuration
                    print(
                        f"DEBUG_CHECK: Logger name='{logger.name}', Effective level={logger.getEffectiveLevel()}, Handler count={len(logger.handlers)}, Root handler count={len(__import__('logging').getLogger().handlers)}"
                    )
                    first_empty_shards.append(f"{i + 1} ({traj_file})")
                continue

            # Log shape immediately after loading
            logger.info(
                "Trajectory %d (%s): shape = (%d frames, %d atoms)",
                i + 1,
                traj_file,
                joined.n_frames,
                joined.n_atoms,
            )

            self.trajectories.append(joined)
            logger.info("Loaded trajectory %d: %d frames", i + 1, joined.n_frames)
            self._maybe_load_demux_metadata(Path(traj_file))

        # Log summary of skipped shards
        if skipped_shard_count > 0 or empty_shard_count > 0:
            logger.warning(
                "Shard loading summary: %d total shards processed, %d skipped (failed/None), %d empty (0 frames)",
                total_shards_processed,
                skipped_shard_count,
                empty_shard_count,
            )
            if skipped_shard_count > MAX_DETAILED_LOGS:
                logger.warning(
                    "First %d skipped shards: %s (... and %d more)",
                    MAX_DETAILED_LOGS,
                    first_skipped_shards,
                    skipped_shard_count - MAX_DETAILED_LOGS,
                )
            elif first_skipped_shards:
                logger.warning("Skipped shards: %s", first_skipped_shards)

            if empty_shard_count > MAX_DETAILED_LOGS:
                logger.warning(
                    "First %d empty shards: %s (... and %d more)",
                    MAX_DETAILED_LOGS,
                    first_empty_shards,
                    empty_shard_count - MAX_DETAILED_LOGS,
                )
            elif first_empty_shards:
                logger.warning("Empty shards: %s", first_empty_shards)

        if not self.trajectories:
            if ignore_errors:
                logger.error(
                    "No trajectories could be loaded; continuing with empty dataset"
                )
                self._update_total_frames()
                return
            raise ValueError("No trajectories loaded successfully")

        logger.info(f"Total trajectories loaded: {len(self.trajectories)}")
        self._update_total_frames()

    def _resolve_atom_indices(
        self, atom_selection: str | Sequence[int] | None
    ) -> Sequence[int] | None:
        if atom_selection is None:
            return None
        topo_path = self._resolve_topology_path()
        topo = load_mdtraj_topology(topo_path)
        logger = getattr(self, "logger", None)
        resolved = resolve_atom_selection(
            topo,
            atom_selection,
            logger=logger,
            on_error="warn",
        )
        if resolved is None:
            return None
        return tuple(int(idx) for idx in cast(Sequence[int], resolved))

    def _resolve_topology_path(self):
        return resolve_project_path(self.topology_file)

    def _stream_single_trajectory(
        self,
        *,
        traj_file: str,
        stride: int,
        atom_indices: Sequence[int] | None,
        chunk_size: int,
        selection_str: str | None,
    ) -> md.Trajectory | None:
        from pmarlo.io import trajectory as traj_io

        resolved_traj = resolve_project_path(traj_file)
        path = Path(resolved_traj)
        if not path.exists():
            import logging as _logging

            _logging.getLogger("pmarlo").warning(
                f"Trajectory file not found: {traj_file}"
            )
            return None
        import logging as _logging

        _logging.getLogger("pmarlo").info(
            "Streaming trajectory %s with stride=%d, chunk=%d%s",
            resolved_traj,
            stride,
            chunk_size,
            f", selection={selection_str}" if selection_str else "",
        )
        joined: md.Trajectory | None = None
        topo_path = resolve_project_path(self.topology_file)

        try:
            for chunk in traj_io.iterload(
                resolved_traj,
                top=topo_path,
                stride=stride,
                atom_indices=atom_indices,
                chunk=chunk_size,
            ):
                joined = chunk if joined is None else joined.join(chunk)
        except Exception as exc:
            if getattr(self, "ignore_trajectory_errors", False):
                _logging.getLogger("pmarlo").error(
                    "Failed to read trajectory %s: %s", resolved_traj, exc
                )
                return None
            raise
        if joined is None:
            _logging.getLogger("pmarlo").warning(f"No frames loaded from {traj_file}")
        return joined

    def _maybe_load_demux_metadata(self, traj_path: Path) -> None:
        if getattr(self, "demux_metadata", None) is not None:
            return
        meta_path = traj_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return
        try:
            from pmarlo.demultiplexing.demux_metadata import DemuxMetadata

            meta = DemuxMetadata.from_json(meta_path)
            self.demux_metadata = meta
            stride_frames = meta.exchange_frequency_steps // meta.frames_per_segment
            self.frame_stride = stride_frames
            self.time_per_frame_ps = meta.integration_timestep_ps * stride_frames
            import logging as _logging

            _logging.getLogger("pmarlo").info(
                "Loaded demux metadata: stride=%d, dt=%.4f ps",
                stride_frames,
                self.time_per_frame_ps,
            )
        except Exception as exc:  # pragma: no cover - defensive
            import logging as _logging

            _logging.getLogger("pmarlo").warning(
                f"Failed to parse metadata {meta_path}: {exc}"
            )
