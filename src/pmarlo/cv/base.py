from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class CVModel(Protocol):
    """Protocol for learned collective variable models.

    Implementations should be lightweight wrappers that can:
    - transform feature matrices into CVs
    - persist their state and reload it deterministically
    - export to TorchScript for PLUMED interoperability
    - emit a minimal PLUMED snippet referencing the exported model
    """

    def transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - Protocol
        ...

    def save(self, path: Path) -> None:  # pragma: no cover - Protocol
        ...

    @classmethod
    def load(cls, path: Path) -> "CVModel":  # pragma: no cover - Protocol
        ...

    def to_torchscript(self, path: Path) -> Path:  # pragma: no cover - Protocol
        ...

    def plumed_snippet(self, model_path: Path) -> str:  # pragma: no cover - Protocol
        ...
