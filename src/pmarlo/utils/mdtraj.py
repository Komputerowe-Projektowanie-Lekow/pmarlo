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

    if on_error not in {"raise", "warn", "ignore"}:
        raise ValueError("on_error must be 'raise', 'warn', or 'ignore'")

    def _handle_failure(exc: Exception | None) -> None:
        if on_error == "raise":
            if exc is None:
                raise ValueError("atom selection produced no atoms")
            raise exc
        if on_error == "warn" and logger is not None:
            msg = "atom selection failed"
            logger.warning(msg if exc is None else f"{msg}: {exc}")

    if isinstance(atom_selection, str):
        try:
            selection = topo.select(atom_selection)
        except Exception as exc:  # pragma: no cover - delegated to mdtraj
            _handle_failure(exc)
            return None
        if selection.size == 0:
            _handle_failure(None)
            return None
        return [int(i) for i in selection]

    try:
        selection = [int(i) for i in atom_selection]
    except Exception as exc:
        _handle_failure(exc)
        return None
    if not selection:
        _handle_failure(None)
        return None
    return selection
