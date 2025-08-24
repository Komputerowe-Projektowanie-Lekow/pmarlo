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

    gen = md.iterload(
        filename,
        top=top,
        stride=stride,
        atom_indices=atom_indices,
        chunk=chunk,
    )

    if verbose_plugin_logs:
        try:
            for chunk_traj in gen:
                yield chunk_traj
        finally:
            gen.close()
        return

    with _suppress_plugin_output():
        try:
            for chunk_traj in gen:
                yield chunk_traj
        finally:
            gen.close()
