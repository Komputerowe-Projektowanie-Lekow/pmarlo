"""Trajectory I/O helpers with quiet plugin logging.

This module wraps :mod:`mdtraj` trajectory loaders to silence the noisy
VMD DCD plugin that prints diagnostic information directly to stdout.
Users can opt into verbose plugin logs by setting
:data:`pmarlo.io.verbose_plugin_logs` to ``True``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from typing import Iterator, Sequence

from . import verbose_plugin_logs

if verbose_plugin_logs:
    import mdtraj as md  # type: ignore
else:  # pragma: no cover - import side effect only
    with open(os.devnull, "w") as devnull:
        fd_out, fd_err = os.dup(1), os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            import mdtraj as md  # type: ignore
        finally:
            os.dup2(fd_out, 1)
            os.dup2(fd_err, 2)
            os.close(fd_out)
            os.close(fd_err)

_LOGGERS = ["mdtraj.formats.registry", "mdtraj.formats.dcd"]


@contextlib.contextmanager
def _suppress_plugin_output() -> Iterator[None]:
    """Temporarily silence mdtraj's DCD plugin noise.

    This redirects C-level prints to ``stdout``/``stderr`` and downgrades
    the relevant Python loggers to ``WARNING`` for the duration of the
    context, restoring previous levels afterwards.
    """

    if verbose_plugin_logs:
        # Nothing to do; yield control immediately.
        yield
        return

    # Store previous logger levels to restore later
    prev_levels = {}
    for name in _LOGGERS:
        logger = logging.getLogger(name)
        prev_levels[name] = logger.level
        logger.setLevel(logging.WARNING)

    # Redirect low-level file descriptors to devnull to silence C prints
    with open(os.devnull, "w") as devnull:
        fd_out, fd_err = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # pragma: no cover
            pass
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:  # pragma: no cover
                pass
            os.dup2(fd_out, 1)
            os.dup2(fd_err, 2)
            os.close(fd_out)
            os.close(fd_err)
            for name, level in prev_levels.items():
                logging.getLogger(name).setLevel(level)


def _resolve_path(path: str | None) -> str | None:
    """Resolve relative paths robustly for tests and different CWDs."""
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    alt = os.path.join(os.getcwd(), path)
    if os.path.exists(alt):
        return alt
    try:
        from pathlib import Path as _Path

        # Repository root is three levels up from this file: io -> pmarlo -> src -> repo
        root = _Path(__file__).resolve().parents[3]
        candidate = root / path.replace("/", os.sep)
        if candidate.exists():
            return str(candidate)
    except Exception:  # pragma: no cover
        pass
    return path


def _make_iterload_generator(
    filename: str,
    *,
    top: str | md.Trajectory | None,
    stride: int,
    atom_indices: Sequence[int] | None,
    chunk: int,
):
    return md.iterload(
        _resolve_path(filename) or filename,
        top=_resolve_path(top) if isinstance(top, str) else top,
        stride=stride,
        atom_indices=atom_indices,
        chunk=chunk,
    )


def _yield_frames_plain(gen) -> Iterator[md.Trajectory]:
    try:
        for chunk_traj in gen:
            yield chunk_traj
    finally:
        gen.close()


def _yield_frames_with_logging(
    gen, *, chunk: int, stride: int, logger: logging.Logger
) -> Iterator[md.Trajectory]:
    try:
        total = 0
        for chunk_traj in gen:
            total += int(getattr(chunk_traj, "n_frames", 0))
            if total % max(1, chunk) == 0:
                logger.info(
                    "[iterload] streamed %d frames (chunk=%d, stride=%d)",
                    total,
                    int(chunk),
                    int(stride),
                )
            yield chunk_traj
    finally:
        gen.close()


def iterload(
    filename: str,
    *,
    top: str | md.Trajectory | None = None,
    stride: int = 1,
    atom_indices: Sequence[int] | None = None,
    chunk: int = 1000,
) -> Iterator[md.Trajectory]:
    """Stream trajectory frames quietly from disk.

    Parameters
    ----------
    filename:
        Path to the trajectory file (e.g. DCD).
    top:
        Topology information required by :func:`md.iterload`.
    stride:
        Only return every ``stride``-th frame.
    atom_indices:
        Optional subset of atoms to load.
    chunk:
        Number of frames to yield per iteration.
    """

    logger = logging.getLogger("pmarlo")

    gen = _make_iterload_generator(
        filename,
        top=top,
        stride=stride,
        atom_indices=atom_indices,
        chunk=chunk,
    )

    if verbose_plugin_logs:
        yield from _yield_frames_plain(gen)
        return

    with _suppress_plugin_output():
        yield from _yield_frames_with_logging(
            gen, chunk=int(chunk), stride=int(stride), logger=logger
        )
