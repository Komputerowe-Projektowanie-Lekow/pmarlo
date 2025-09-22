"""Legacy shim that forwards to :mod:`pmarlo.devtools.lines_report`."""

from __future__ import annotations

import warnings

from pmarlo.devtools.lines_report import main as _main

_DEPRECATION_MESSAGE = (
    "utilities/lines.py is deprecated; use the 'pmarlo-lines-report' console"
    " script instead."
)


def main() -> int:
    """Execute the modern lines report while emitting a deprecation warning."""

    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    return _main()


if __name__ == "__main__":  # pragma: no cover - compatibility shim
    raise SystemExit(main())
