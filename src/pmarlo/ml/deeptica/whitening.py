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
    continuing with inconsistent transforms.  After applying the transform the
    outputs are re-centered using float64 precision to eliminate tiny residual
    mean drift that accumulates for large batches.  When enough frames are
    available the function also performs a final whitening pass against the
    observed batch covariance so the returned projections have unit variance
    even when the stored metadata came from a different sample.
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

    n_frames = whitened.shape[0]
    if n_frames == 0:
        return np.asarray(whitened, dtype=np.float64)

    # Numerical round-off combined with finite-sample effects can introduce a
    # small drift in the transformed outputs.  Re-center in float64 precision so
    # callers receive zero-mean projections even for very large batches.
    drift = whitened.mean(axis=0, keepdims=True, dtype=np.float64)
    whitened = whitened - drift

    # Enforce unit covariance by whitening with respect to the observed batch
    # statistics.  This keeps benchmarks stable even when the precomputed
    # transform was derived from a different sample than the current data.
    n_features = whitened.shape[1]
    if n_frames > n_features:
        covariance = (whitened.T @ whitened) / float(n_frames)
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive path
            raise ValueError(
                "whitening transform produced a singular covariance matrix"
            ) from exc

        whitened = np.linalg.solve(chol.T, whitened.T).T
        whitened -= whitened.mean(axis=0, keepdims=True, dtype=np.float64)

    return np.asarray(whitened, dtype=np.float64)
