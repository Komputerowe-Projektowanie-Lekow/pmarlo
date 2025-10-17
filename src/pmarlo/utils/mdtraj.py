from __future__ import annotations

"""Helpers for working with :mod:`mdtraj` dependencies."""

import logging
import os
from pathlib import Path
from typing import Literal, Sequence

import mdtraj as md


__all__ = ["load_mdtraj_topology", "resolve_atom_selection"]


def _as_str_path(topology: str | os.PathLike[str] | Path) -> str:
    """Return a string representation suitable for mdtraj loading."""
    return str(Path(topology))


def load_mdtraj_topology(topology: str | os.PathLike[str] | Path) -> "md.Topology":
    """Load an MDTraj topology from ``topology``."""

    return md.load_topology(_as_str_path(topology))


def resolve_atom_selection(
    topo: "md.Topology",
    atom_selection: str | Sequence[int] | None,
    *,
    logger: logging.Logger | None = None,
    on_error: Literal["raise", "warn", "ignore"] = "raise",
) -> Sequence[int] | None:
    """Resolve an atom selection against ``topo``."""

    if atom_selection is None:
        return None

    if on_error != "raise":
        raise ValueError("on_error must be 'raise'")

    if isinstance(atom_selection, str):
        selection = topo.select(atom_selection)
        return [int(i) for i in selection]

    return [int(i) for i in atom_selection]
