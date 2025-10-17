from __future__ import annotations

# 'Optional' was unused; keep imports minimal to satisfy flake8

from openmm.app import DCDReporter as _DCDReporter


class ClosableDCDReporter(_DCDReporter):
    """DCDReporter with a public close() for safe file finalization.

    OpenMM's reporter stores a private file handle. Accessing it across
    versions is fragile; this wrapper exposes a best-effort close().
    """

    def __init__(self, file: str, reportInterval: int):
        super().__init__(file, reportInterval)

    def close(self) -> None:
        if not hasattr(self, "_out"):
            raise AttributeError("OpenMM DCDReporter no longer exposes '_out'")
        handle = getattr(self, "_out")
        if handle is None:
            raise RuntimeError("ClosableDCDReporter._out is None; file already closed")
        handle.close()
