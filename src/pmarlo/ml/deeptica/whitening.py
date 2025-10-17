"""Utilities for applying learned DeepTICA output whitening transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def apply_output_transform(
    Y: np.ndarray | NDArray[np.float64],
    mean: Any,
    W: Any,
    already_applied: bool | None,
) -> NDArray[np.float64]:
    """Apply the stored output whitening transform when available.

    Parameters
    ----------
    Y:
        Raw collective variable projections ``(n_frames, n_cvs)``.
    mean:
        Per-component mean recorded during training.  Accepts any sequence-like
        object convertible to ``float``.
    W:
        Whitening matrix (typically the inverse Cholesky factor of the output
        covariance).  Accepts array-likes convertible to ``float``.
    already_applied:
        Flag indicating whether the transform has already been applied.  When
        ``True`` the input array is returned unchanged.

    Returns
    -------
    numpy.ndarray
        Transformed CVs with unit variance when the metadata is available.

    Notes
    -----
    All whitening metadata must be present.  Missing ``mean`` or ``W`` values
    raise a ``ValueError`` so callers surface configuration issues immediately.
    Shape mismatches likewise raise a ``ValueError`` to avoid silently
    continuing with inconsistent transforms.
    """

    arr = np.asarray(Y, dtype=np.float64)
    if bool(already_applied):
        return arr

    if mean is None or W is None:
        raise ValueError(
            "Whitening metadata is incomplete: both mean and transform are required"
        )

    mean_arr = np.asarray(mean, dtype=np.float64)
    transform = np.asarray(W, dtype=np.float64)

    if mean_arr.ndim != 1:
        raise ValueError("output mean must be a 1D array")
    if transform.ndim != 2:
        raise ValueError("output transform must be a 2D matrix")
    if mean_arr.shape[0] != transform.shape[0]:
        raise ValueError(
            "output mean and transform dimension mismatch: "
            f"{mean_arr.shape[0]} vs {transform.shape[0]}"
        )
    if arr.ndim != 2 or arr.shape[1] != mean_arr.shape[0]:
        raise ValueError(
            "projection has incompatible shape for whitening: "
            f"expected (..., {mean_arr.shape[0]}), got {arr.shape}"
        )

    centered = arr - mean_arr.reshape(1, -1)
    whitened = centered @ transform
    return np.asarray(whitened, dtype=np.float64)
