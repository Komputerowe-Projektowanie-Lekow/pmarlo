"""Free energy surface result containers and utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class FESResult:
    """Container for free energy surface data.

    Attributes
    ----------
    F : np.ndarray
        2D array of free energy values.
    xedges : np.ndarray
        Bin edges for the first collective variable.
    yedges : np.ndarray
        Bin edges for the second collective variable.
    levels_kJmol : np.ndarray | None
        Optional contour levels in kJ/mol for plotting convenience.
    metadata : dict[str, Any]
        Additional information about the FES calculation.
    """

    F: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    levels_kJmol: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _mapping = {
        "F": "F",
        "xedges": "xedges",
        "yedges": "yedges",
        "levels_kJmol": "levels_kJmol",
        "metadata": "metadata",
    }

    def __getitem__(self, key: str) -> Any:  # pragma: no cover - backward compat
        """Provide dict-style access with a deprecation warning.

        Parameters
        ----------
        key: str
            Attribute name such as ``"F"`` or ``"xedges"``.

        Returns
        -------
        Any
            The corresponding attribute value.
        """

        warnings.warn(
            "Dict-style access to FESResult is deprecated; use attribute access instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if key in self._mapping:
            return getattr(self, self._mapping[key])
        raise KeyError(key)
