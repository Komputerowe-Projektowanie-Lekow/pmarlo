from __future__ import annotations

# 'Optional' was unused; keep imports minimal to satisfy flake8

try:
    # Import lazily, but this file is small and safe to import
    from openmm.app import DCDReporter as _DCDReporter
except Exception:  # pragma: no cover - only occurs when OpenMM missing
    _DCDReporter = object  # type: ignore


class ClosableDCDReporter(_DCDReporter):
    """DCDReporter with a public close() for safe file finalization.

    OpenMM's reporter stores a private file handle. Accessing it across
    versions is fragile; this wrapper exposes a best-effort close().
    """

    def __init__(self, file: str, reportInterval: int):
        super().__init__(file, reportInterval)

    def close(self) -> None:
        try:
            if hasattr(self, "_out") and getattr(self, "_out"):
                self._out.close()  # type: ignore[attr-defined]
        except Exception:
            # Best-effort close; ignore if reporter structure differs
            pass
