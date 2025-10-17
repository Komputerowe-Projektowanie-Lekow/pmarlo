"""DeepTICA feature helpers that require the full optional dependency stack."""

from __future__ import annotations

from . import _full as _full_impl


def _export_public_symbols() -> dict[str, object]:
    exported = getattr(_full_impl, "__all__", None)
    if exported is None:
        exported = [name for name in vars(_full_impl) if not name.startswith("_")]
    return {name: getattr(_full_impl, name) for name in exported}


_namespace = _export_public_symbols()
globals().update(_namespace)
__all__ = list(_namespace)
