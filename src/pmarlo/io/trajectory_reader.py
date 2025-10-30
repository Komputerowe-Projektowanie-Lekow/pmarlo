"""Streaming trajectory readers.

Provides a small I/O abstraction to iterate over only the needed frames
from a trajectory without loading the whole file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol

import numpy as np

logger = logging.getLogger("pmarlo")


class TrajectoryIOError(Exception):
    """Base error for trajectory I/O issues."""


class TrajectoryFormatError(TrajectoryIOError):
    """Raised when the trajectory format cannot be read or is corrupted."""


class TrajectoryMissingTopologyError(TrajectoryIOError):
    """Raised when topology is required by the format but not provided."""


class TrajectoryReader(Protocol):
    """Protocol for minimal streaming readers.

    Implementations should not load entire files into memory and must yield
    per-frame coordinate arrays of shape ``(n_atoms, 3)``.
    """

    def iter_frames(
        self, path: str, start: int, stop: int, stride: int = 1
    ) -> Iterator[np.ndarray]:
        """Iterate frames from ``start`` to ``stop`` (exclusive) with ``stride``.

        Parameters
        ----------
        path : str
            Path to the trajectory file.
        start : int
            Start frame index (inclusive). Negative values are treated as 0.
        stop : int
            Stop frame index (exclusive). If larger than the file length, the
            iteration stops at the end of the file.
        stride : int
            Return every ``stride``-th frame in the interval. Values <= 0 are
            coerced to 1.
        """

    def probe_length(self, path: str) -> int:
        """Return the number of frames in the trajectory.

        Implementations should avoid loading the entire file into memory.
        """


@dataclass
class MDAnalysisReader:
    """MDAnalysis-backed streaming reader.

    Parameters
    ----------
    topology_path : str or None
        Required for formats like DCD.
    chunk_size : int
        Ignored for MDAnalysis; kept for API symmetry.
    """

    topology_path: Optional[str] = None
    chunk_size: int = 1000

    def _require(self):
        try:
            pass  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency optional
            raise TrajectoryIOError(
                "MDAnalysis is required for backend='mdanalysis'. Install extra 'pmarlo[mdanalysis]' or 'MDAnalysis'."
            ) from exc

    def iter_frames(
        self, path: str, start: int, stop: int, stride: int = 1
    ) -> Iterator[np.ndarray]:
        self._require()
        import MDAnalysis as mda  # type: ignore

        if self.topology_path is None:
            raise TrajectoryMissingTopologyError(
                f"Topology is required to read '{path}' with MDAnalysis. Provide topology_path."
            )
        start = max(0, int(start))
        stop = max(start, int(stop))
        stride = 1 if stride <= 0 else int(stride)
        try:
            u = mda.Universe(self.topology_path, path)
            sel = slice(start, stop, stride)
            for ts in u.trajectory[sel]:
                # MDAnalysis uses Angstroms by default; we return raw arrays to avoid assumptions
                yield np.array(ts.positions, copy=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise TrajectoryIOError(
                f"Failed MDAnalysis stream of '{path}': {exc}"
            ) from exc

    def probe_length(self, path: str) -> int:
        self._require()
        import MDAnalysis as mda  # type: ignore

        if self.topology_path is None:
            raise TrajectoryMissingTopologyError(
                f"Topology is required to read '{path}' with MDAnalysis. Provide topology_path."
            )
        try:
            u = mda.Universe(self.topology_path, path)
            return int(len(u.trajectory))
        except Exception as exc:  # pragma: no cover - defensive
            raise TrajectoryIOError(
                f"Failed to probe length with MDAnalysis for '{path}': {exc}"
            ) from exc


def get_reader(backend: str, topology_path: Optional[str]) -> TrajectoryReader:
    """Return a reader instance for the requested backend.

    Parameters
    ----------
    backend : {"mdtraj", "mdanalysis"}
        Reader backend to use. ``"mdtraj"`` is the default.
    topology_path : str or None
        Path to the topology file when required by the format (e.g., DCD).

    Returns
    -------
    TrajectoryReader
        Reader instance bound to the requested backend.

    Raises
    ------
    TrajectoryIOError
        If the backend is unknown or MDAnalysis is requested but not installed.

    Examples
    --------
    >>> reader = get_reader("mdtraj", topology_path="model.pdb")
    >>> isinstance(reader, MDTrajReader)
    True
    """
    backend = (backend or "mdtraj").lower()
    if backend == "mdtraj":
        return MDTrajReader(topology_path=topology_path)
    if backend == "mdanalysis":
        try:
            import MDAnalysis  # noqa: F401
        except Exception as exc:
            raise TrajectoryIOError(
                "MDAnalysis backend selected but 'MDAnalysis' is not installed."
            ) from exc
        return MDAnalysisReader(topology_path=topology_path)
    raise TrajectoryIOError(f"Unknown trajectory reader backend: {backend}")


@dataclass
class MDTrajReader:
    """MDTraj-backed streaming reader.

    Parameters
    ----------
    topology_path : str or None, optional
        Path to topology for formats that require it (e.g., DCD). If omitted,
        the reader will try to proceed for self-contained formats, otherwise it
        raises :class:`TrajectoryMissingTopologyError`.
    chunk_size : int, optional
        Number of frames to read per chunk when streaming.
    """

    topology_path: Optional[str] = None
    chunk_size: int = 1000

    def _requires_topology(self, path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in {".dcd", ".xtc", ".trr", ".nc"}

    def _iterload(self, path: str, *, stride: int = 1):  # type: ignore[override]
        """Open an iterator over trajectory chunks with plugin chatter silenced.

        Uses pmarlo.io.trajectory.iterload which redirects the VMD/MDTraj DCD
        plugin output away from stdout/stderr, preventing noisy console spam.
        """
        top = self.topology_path
        if self._requires_topology(path) and not top:
            raise TrajectoryMissingTopologyError(
                f"Topology is required to read '{path}'. Provide topology_path."
            )

        try:
            from pmarlo.io import trajectory as _traj_io

            return _traj_io.iterload(
                path, top=top, chunk=int(self.chunk_size), stride=1
            )
        except TrajectoryIOError:
            raise
        except Exception as exc:
            raise TrajectoryFormatError(
                f"Failed to open trajectory '{path}': {exc}"
            ) from exc

    def iter_frames(
        self, path: str, start: int, stop: int, stride: int = 1
    ) -> Iterator[np.ndarray]:
        """Iterate per-frame coordinates between ``start`` and ``stop``.

        This implementation streams the trajectory sequentially using
        ``mdtraj.iterload`` and filters frames by index. It never joins chunks
        into a single in-memory trajectory.
        """
        if stride <= 0:
            logger.warning("iter_frames called with stride <= 0; coercing to 1")
            stride = 1
        start = max(0, int(start))
        stop = max(start, int(stop))

        logger.info(
            "Streaming frames: path=%s start=%d stop=%d stride=%d chunk=%d",
            path,
            start,
            stop,
            stride,
            self.chunk_size,
        )

        global_index = 0
        try:
            for chunk in self._iterload(path, stride=1):
                # chunk.xyz: (n_frames, n_atoms, 3)
                n = int(chunk.n_frames)
                # Fast skip over pre-start frames by advancing the global index
                chunk_start = global_index
                chunk_stop = global_index + n
                if chunk_stop <= start:
                    global_index += n
                    continue

                # Determine local frame indices within the chunk to yield
                local_start = max(0, start - chunk_start)
                local_stop = min(n, stop - chunk_start)
                for local_i in range(local_start, local_stop):
                    global_i = chunk_start + local_i
                    if (global_i - start) % stride != 0:
                        continue
                    # Return a copy to ensure independence from chunk lifetime
                    yield np.array(chunk.xyz[local_i], copy=True)
                global_index += n
                if global_index >= stop:
                    break
        except TrajectoryIOError:
            raise
        except Exception as exc:
            raise TrajectoryIOError(
                f"Error while streaming frames from '{path}': {exc}"
            ) from exc

    def probe_length(self, path: str) -> int:
        """Return the number of frames by streaming and counting.

        Uses sequential iteration to avoid loading entire files into memory.
        """
        total = 0
        try:
            for chunk in self._iterload(path, stride=1):
                total += int(chunk.n_frames)
            return total
        except TrajectoryIOError:
            raise
        except Exception as exc:
            raise TrajectoryIOError(
                f"Failed to probe length of '{path}': {exc}"
            ) from exc
