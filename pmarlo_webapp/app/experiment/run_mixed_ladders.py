from __future__ import annotations

import sys
from typing import Sequence

from .runner import run_experiment_cli


def main(argv: Sequence[str] | None = None) -> int:
    return run_experiment_cli("E1_mixed_ladders", argv)


if __name__ == "__main__":
    raise SystemExit(main())
