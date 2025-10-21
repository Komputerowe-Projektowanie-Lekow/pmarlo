from __future__ import annotations

"""Helpers for working with :mod:`mdtraj` dependencies."""

import logging
import os
from pathlib import Path
from typing import Callable, Literal, Sequence

import mdtraj as md

__all__ = ["load_mdtraj_topology", "resolve_atom_selection"]


def _as_str_path(topology: str | os.PathLike[str] | Path) -> str:
    """Return a string representation suitable for mdtraj loading."""
    return str(Path(topology))


def load_mdtraj_topology(topology: str | os.PathLike[str] | Path) -> "md.Topology":
    """Load an MDTraj topology from ``topology``."""

    return md.load_topology(_as_str_path(topology))


def _validate_on_error(on_error: str) -> None:
    if on_error not in {"raise", "warn", "ignore"}:
        raise ValueError("on_error must be 'raise', 'warn', or 'ignore'")


def _resolve_selection_from_string(
    topo: "md.Topology",
    expression: str,
    handle_failure: Callable[[Exception | None], None],
) -> Sequence[int] | None:
    try:
        selection = topo.select(expression)
    except (ValueError, TypeError) as exc:  # pragma: no cover - delegated to mdtraj
        handle_failure(exc)
        return None
    if selection.size == 0:
        handle_failure(None)
        return None
    return [int(i) for i in selection]


def _resolve_selection_from_sequence(
    selection: Sequence[int | str],
    handle_failure: Callable[[Exception | None], None],
) -> Sequence[int] | None:
    try:
        indices: list[int] = []
        for item in selection:
            if isinstance(item, str):
                indices.append(int(item, 10))
            else:
                indices.append(int(item))
    except (TypeError, ValueError) as exc:
        handle_failure(exc)
        return None
    if not indices:
        handle_failure(None)
        return None
    return indices


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

    _validate_on_error(on_error)

    def _handle_failure(exc: Exception | None) -> None:
        if on_error == "raise":
            if exc is None:
                raise ValueError("atom selection produced no atoms")
            raise exc
        if on_error == "warn" and logger is not None:
            msg = "atom selection failed"
            logger.warning(msg if exc is None else f"{msg}: {exc}")

    if isinstance(atom_selection, str):
        return _resolve_selection_from_string(topo, atom_selection, _handle_failure)
    return _resolve_selection_from_sequence(atom_selection, _handle_failure)
