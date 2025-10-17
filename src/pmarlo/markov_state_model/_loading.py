from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

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
                continue
            self.trajectories.append(joined)
            logger.info("Loaded trajectory %d: %d frames", i + 1, joined.n_frames)
            self._maybe_load_demux_metadata(Path(traj_file))

        if not self.trajectories:
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
        return resolve_atom_selection(
            topo,
            atom_selection,
            logger=logger,
            on_error="warn",
        )

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

        for chunk in traj_io.iterload(
            resolved_traj,
            top=topo_path,
            stride=stride,
            atom_indices=atom_indices,
            chunk=chunk_size,
        ):
            joined = chunk if joined is None else joined.join(chunk)
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
