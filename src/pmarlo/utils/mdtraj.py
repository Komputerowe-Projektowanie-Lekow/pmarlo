from __future__ import annotations

"""Helpers for working with optional :mod:`mdtraj` dependencies."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    import mdtraj as md


__all__ = ["load_mdtraj_topology", "resolve_atom_selection"]


def _as_str_path(topology: str | os.PathLike[str] | Path) -> str:
    """Return a string representation suitable for mdtraj loading."""
    return str(Path(topology))


def load_mdtraj_topology(topology: str | os.PathLike[str] | Path) -> "md.Topology":
    """Load an MDTraj topology from ``topology``.

    The import of :mod:`mdtraj` happens lazily so callers can depend on this
    utility without importing optional runtime dependencies during module
    import.
    """

    import mdtraj as md  # type: ignore

    return md.load_topology(_as_str_path(topology))


def resolve_atom_selection(
    topo: "md.Topology",
    atom_selection: str | Sequence[int] | None,
    *,
    logger: logging.Logger | None = None,
    on_error: Literal["raise", "warn", "ignore"] = "raise",
) -> Sequence[int] | None:
    """Resolve an atom selection against ``topo``.

    Parameters
    ----------
    topo:
        The topology to query.
    atom_selection:
        Selection expressed either as an MDTraj DSL string or a sequence of
        integer indices.
    logger:
        Logger used when ``on_error`` is ``"warn"``.
    on_error:
        Behaviour when a string selection fails. Accepted values are ``"raise"``
        (default), ``"warn"`` (log a warning and return ``None``) and
        ``"ignore"`` (silently return ``None``).
    """

    if atom_selection is None:
        return None

    if isinstance(atom_selection, str):
        try:
            selection = topo.select(atom_selection)
        except Exception as exc:  # pragma: no cover - defensive
            if on_error == "warn":
                (logger or logging.getLogger("pmarlo")).warning(
                    "Failed atom selection '%s': %s; using all atoms",
                    atom_selection,
                    exc,
                )
                return None
            if on_error == "ignore":
                return None
            raise
        return [int(i) for i in selection]

    return [int(i) for i in atom_selection]
