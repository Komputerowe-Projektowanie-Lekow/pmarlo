"""Legacy shim that forwards to :mod:`pmarlo.devtools.check_extras_parity`."""

from __future__ import annotations

import warnings

from pmarlo.devtools.check_extras_parity import main as _main

_DEPRECATION_MESSAGE = (
    "tools/check_extras_parity.py is deprecated; use the 'pmarlo-check-extras'"
    " console script instead."
)


def main() -> int:
    """Execute the modern parity checker while emitting a deprecation warning."""

    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    return _main()


if __name__ == "__main__":  # pragma: no cover - compatibility shim
    raise SystemExit(main())
