"""Append-like trajectory writer abstraction with chunked rewrite fallback.

This module provides a small abstraction for incrementally writing frames to a
trajectory. If true streaming append is not safely supported by the backend,
we fall back to a "chunked rewrite" strategy: buffer frames and periodically
rewrite the output file including previously written frames. This is safe and
deterministic, and for large datasets should be paired with sufficiently sized
buffer thresholds and/or a backend that supports streaming writes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Self

import numpy as np

from .trajectory_reader import MDTrajReader


@dataclass
class MDAnalysisDCDWriter:
    """MDAnalysis-based DCD writer with simple append-like API.

    This writer opens a DCDWriter and streams frames one-by-one. It is not an
    in-place appender; repeated calls write frames sequentially to the same file
    handle.
    """

    topology_path: Optional[str] = None
    _path: Optional[str] = field(default=None, init=False, repr=False)
    _n_atoms: Optional[int] = field(default=None, init=False, repr=False)
    _writer: Optional[object] = field(default=None, init=False, repr=False)
    _is_open: bool = field(default=False, init=False, repr=False)

    def _require(self):
        try:
            pass  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency optional
            raise TrajectoryWriteError(
                "MDAnalysis is required for backend='mdanalysis'. Install extra 'pmarlo[mdanalysis]' or 'MDAnalysis'."
            ) from exc

    def open(
        self, path: str, topology_path: str | None, overwrite: bool = False
    ) -> "MDAnalysisDCDWriter":
        self._require()
        if self._is_open:
            raise TrajectoryWriteError("Writer is already open")
        self._path = str(path)
        self.topology_path = topology_path or self.topology_path
        p = Path(self._path)
        if p.exists() and not overwrite:
            raise TrajectoryWriteError(
                f"Output file exists and overwrite=False: {path}"
            )
        if p.exists():
            try:
                p.unlink()
            except Exception as exc:
                raise TrajectoryWriteError(
                    f"Failed to overwrite existing file: {exc}"
                ) from exc

        # Writer created on first write when n_atoms is known
        self._n_atoms = None
        self._writer = None
        self._is_open = True
        return self

    def write_frames(self, coords: np.ndarray, box: np.ndarray | None = None) -> None:
        self._require()
        if not self._is_open or self._path is None:
            raise TrajectoryWriteError("Writer is not open")
        c = np.asarray(coords)
        if c.ndim != 3 or c.shape[-1] != 3:
            raise TrajectoryWriteError(
                f"coords must have shape (n_frames, n_atoms, 3); got {c.shape}"
            )
        if c.size == 0:
            return
        from MDAnalysis.coordinates.DCD import DCDWriter  # type: ignore

        if self._n_atoms is None:
            self._n_atoms = int(c.shape[1])
            self._writer = DCDWriter(self._path, n_atoms=self._n_atoms)
        elif int(c.shape[1]) != int(self._n_atoms):
            raise TrajectoryWriteError(
                f"Inconsistent atom count: expected {self._n_atoms}, got {c.shape[1]}"
            )
        writer = self._writer
        assert writer is not None
        # MDAnalysis DCD expects Angstroms; we write raw floats as-is to avoid implicit unit conversions
        for i in range(c.shape[0]):
            writer.write_next_timestep(c[i, :, :])  # type: ignore[arg-type,attr-defined]

    def close(self) -> None:
        if not self._is_open:
            return
        try:
            if self._writer is not None:
                try:
                    self._writer.close()  # type: ignore[call-arg,attr-defined]
                except Exception:
                    pass
        finally:
            self._is_open = False

    def flush(self) -> None:
        # MDAnalysis DCDWriter does not expose flush; best-effort no-op
        try:
            if self._writer is not None and hasattr(self._writer, "flush"):
                self._writer.flush()  # type: ignore[call-arg]
        except Exception:
            pass


def get_writer(backend: str, topology_path: Optional[str]) -> TrajectoryWriter:
    """Return a writer instance for the requested backend."""
    backend = (backend or "mdtraj").lower()
    if backend == "mdtraj":
        return MDTrajDCDWriter()
    if backend == "mdanalysis":
        try:
            import MDAnalysis  # noqa: F401
        except Exception as exc:
            raise TrajectoryWriteError(
                "MDAnalysis backend selected but 'MDAnalysis' is not installed."
            ) from exc
        return MDAnalysisDCDWriter(topology_path=topology_path)
    raise TrajectoryWriteError(f"Unknown trajectory writer backend: {backend}")


logger = logging.getLogger("pmarlo")


class TrajectoryWriteError(Exception):
    """Base error for trajectory writing issues."""


class TrajectoryWriter(Protocol):
    """Protocol for minimal append-like writers.

    Implementations write frames incrementally without requiring all frames in
    memory.
    """

    def open(
        self, path: str, topology_path: str | None, overwrite: bool = False
    ) -> Self:  # noqa: D401 - short doc
        """Open or create an output trajectory at ``path``.

        Parameters
        ----------
        path : str
            Output trajectory file path (e.g., ``"out.dcd"``).
        topology_path : str or None
            Topology required by some formats (e.g., DCD). If ``None`` and the
            format requires it, an error is raised.
        overwrite : bool, optional
            If True, replace any existing file at ``path``; otherwise raise if
            the file exists (default False).
        """

    def write_frames(
        self, coords: np.ndarray, box: np.ndarray | None = None
    ) -> None:  # noqa: D401 - short doc
        """Write one or more frames.

        Parameters
        ----------
        coords : ndarray, shape (n_frames, n_atoms, 3)
            Cartesian coordinates for one or more frames.
        box : ndarray or None, optional
            Reserved for future use to pass box vectors per frame; currently
            unused for DCD backends (default None).
        """

    def close(self) -> None:  # noqa: D401 - short doc
        """Finalize and close resources."""

    def flush(self) -> None:  # noqa: D401 - short doc
        """Flush any internal buffers to disk if supported."""


@dataclass
class MDTrajDCDWriter:
    """Append-like DCD writer backed by mdtraj with chunked rewrite fallback.

    Parameters
    ----------
    rewrite_threshold : int
        Number of frames to accumulate before performing a rewrite flush.
    topology_path : str | None
        Topology file required by DCD (e.g., PDB). Must be set via ``open`` or
        constructor.
    """

    rewrite_threshold: int = 1000
    topology_path: Optional[str] = None
    _path: Optional[str] = field(default=None, init=False, repr=False)
    _buffer: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _n_atoms: Optional[int] = field(default=None, init=False, repr=False)
    _total_persisted: int = field(default=0, init=False, repr=False)
    _is_open: bool = field(default=False, init=False, repr=False)

    def open(
        self, path: str, topology_path: str | None, overwrite: bool = False
    ) -> Self:
        if self._is_open:
            raise TrajectoryWriteError("Writer is already open")
        self._path = str(path)
        self.topology_path = topology_path or self.topology_path
        if not self.topology_path:
            raise TrajectoryWriteError("DCD writing requires topology_path")
        p = Path(self._path)
        if p.exists():
            if not overwrite:
                raise TrajectoryWriteError(
                    f"Output file exists and overwrite=False: {path}"
                )
            try:
                p.unlink()
            except Exception as exc:  # pragma: no cover - defensive
                raise TrajectoryWriteError(
                    f"Failed to overwrite existing file: {exc}"
                ) from exc
        self._buffer.clear()
        self._n_atoms = None
        self._total_persisted = 0
        self._is_open = True
        logger.info(
            "Opened MDTrajDCDWriter: %s (threshold=%d)",
            self._path,
            int(self.rewrite_threshold),
        )
        return self

    # Internal helpers
    def _ensure_open(self) -> None:
        if not self._is_open or self._path is None:
            raise TrajectoryWriteError("Writer is not open")

    def _validate_coords(self, coords: np.ndarray) -> np.ndarray:
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise TrajectoryWriteError(
                f"coords must have shape (n_frames, n_atoms, 3); got {coords.shape}"
            )
        if self._n_atoms is None:
            self._n_atoms = int(coords.shape[1])
        elif int(coords.shape[1]) != int(self._n_atoms):
            raise TrajectoryWriteError(
                f"Inconsistent atom count: expected {self._n_atoms}, got {coords.shape[1]}"
            )
        return coords

    def write_frames(self, coords: np.ndarray, box: np.ndarray | None = None) -> None:
        self._ensure_open()
        c = self._validate_coords(np.asarray(coords))
        if c.size == 0:
            return
        # store a copy to decouple from caller's buffer
        self._buffer.append(np.array(c, copy=True))
        buf_frames = sum(arr.shape[0] for arr in self._buffer)
        if buf_frames >= int(self.rewrite_threshold):
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        self._ensure_open()
        assert self._path is not None

        new_chunk = self._consume_buffer()
        total_new = int(new_chunk.shape[0])
        if self._should_write_directly():
            self._rewrite_all(new_chunk)
            self._total_persisted += total_new
            return

        tmp_path = str(Path(self._path).with_suffix(".tmp.dcd"))
        try:
            self._rewrite_with_append(new_chunk, total_new, tmp_path)
        except Exception as exc:  # pragma: no cover - defensive
            self._cleanup_failed_flush(tmp_path)
            raise TrajectoryWriteError(f"Failed to flush buffer: {exc}") from exc

    def _rewrite_all(self, coords: np.ndarray) -> None:
        assert self._path is not None
        try:
            import mdtraj as md  # type: ignore

            topo = md.load_topology(self.topology_path)
            traj = md.Trajectory(coords, topo)
            traj.save_dcd(self._path)
        except Exception as exc:  # pragma: no cover - defensive
            raise TrajectoryWriteError(f"Failed to write DCD: {exc}") from exc

    def _consume_buffer(self) -> np.ndarray:
        new_chunk = np.concatenate(self._buffer, axis=0)
        self._buffer.clear()
        return new_chunk

    def _should_write_directly(self) -> bool:
        assert self._path is not None
        return self._total_persisted == 0 and not Path(self._path).exists()

    def _rewrite_with_append(
        self, new_chunk: np.ndarray, total_new: int, tmp_path: str
    ) -> None:
        assert self._path is not None
        import mdtraj as md  # type: ignore

        reader = MDTrajReader(topology_path=self.topology_path)
        old_path = str(self._path)
        old_len = reader.probe_length(old_path)
        topo = md.load_topology(self.topology_path)
        joined = self._stream_existing_frames(reader, old_path, old_len, topo)
        new_traj = md.Trajectory(new_chunk, topo)
        final = self._join_trajectories(joined, new_traj)
        final.save_dcd(tmp_path)
        Path(tmp_path).replace(self._path)
        self._total_persisted = old_len + total_new

    def _stream_existing_frames(
        self,
        reader: MDTrajReader,
        old_path: str,
        old_len: int,
        topo: "md.Topology",
    ) -> Optional["md.Trajectory"]:
        import mdtraj as md  # type: ignore

        if old_len <= 0:
            return None
        chunk_list: list[md.Trajectory] = []
        joined: Optional[md.Trajectory] = None
        for xyz in reader.iter_frames(old_path, start=0, stop=old_len, stride=1):
            chunk_list.append(md.Trajectory(xyz[np.newaxis, ...], topo))
            if len(chunk_list) >= 256:
                joined = self._join_trajectories(joined, md.join(chunk_list))
                chunk_list.clear()
        if chunk_list:
            joined = self._join_trajectories(joined, md.join(chunk_list))
        return joined

    def _join_trajectories(
        self,
        base: Optional["md.Trajectory"],
        addition: "md.Trajectory",
    ) -> "md.Trajectory":
        if base is None:
            return addition
        return base.join(addition)

    def _cleanup_failed_flush(self, tmp_path: str) -> None:
        try:
            Path(tmp_path).unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass

    def close(self) -> None:
        if not self._is_open:
            return
        try:
            # Flush any remaining frames
            self._flush_buffer()
        finally:
            self._is_open = False
            logger.info(
                "Closed MDTrajDCDWriter: %s (frames persisted=%d)",
                self._path,
                int(self._total_persisted),
            )

    def flush(self) -> None:
        """Force a buffer flush and atomic file replace if needed."""
        if not self._is_open:
            return
        self._flush_buffer()
