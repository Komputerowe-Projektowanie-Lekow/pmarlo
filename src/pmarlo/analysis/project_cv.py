"""Helpers for working with projected collective variables."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Tuple

import numpy as np
from numpy.typing import NDArray

MetadataLike = Mapping[str, Any] | MutableMapping[str, Any]


def apply_whitening_from_metadata(
    values: np.ndarray | NDArray[np.float64], metadata: MetadataLike | None
) -> Tuple[NDArray[np.float64], bool]:
    """Apply the learned output transform described by ``metadata``.

    Parameters
    ----------
    values:
        Projected collective variables.
    metadata:
        Mapping containing ``output_mean``, ``output_transform``, and
        ``output_transform_applied`` fields as produced by the DeepTICA trainer.

    Returns
    -------
    tuple
        A pair ``(whitened, applied)`` where ``whitened`` is the transformed
        array and ``applied`` indicates whether the whitening transform was
        executed for the provided values.
    """

    arr = np.asarray(values, dtype=np.float64)
    if metadata is None:
        raise ValueError("Whitening metadata is required to transform outputs")
    if not isinstance(metadata, Mapping):
        raise TypeError(
            "Whitening metadata must be a mapping with DeepTICA output fields"
        )

    mean = metadata.get("output_mean")
    transform = metadata.get("output_transform")
    already_flag = metadata.get("output_transform_applied")
    if mean is None or transform is None:
        raise ValueError(
            "Whitening metadata must include 'output_mean' and 'output_transform'"
        )

    mean_arr = np.asarray(mean, dtype=np.float64)
    transform_arr = np.asarray(transform, dtype=np.float64)

    if mean_arr.ndim != 1:
        raise ValueError("output mean must be a 1D array")
    if transform_arr.ndim != 2:
        raise ValueError("output transform must be a 2D matrix")
    if transform_arr.shape[0] != mean_arr.shape[0]:
        raise ValueError(
            "output mean and transform dimension mismatch: "
            f"{mean_arr.shape[0]} vs {transform_arr.shape[0]}"
        )
    if arr.ndim != 2 or arr.shape[1] != mean_arr.shape[0]:
        raise ValueError(
            "projection has incompatible shape for whitening: "
            f"expected (..., {mean_arr.shape[0]}), got {arr.shape}"
        )

    if bool(already_flag):
        return np.asarray(arr, dtype=np.float64), False

    centered = arr - mean_arr.reshape(1, -1)
    whitened = centered @ transform_arr

    if isinstance(metadata, MutableMapping):
        metadata["output_transform_applied"] = True  # type: ignore[index]

    return np.asarray(whitened, dtype=np.float64), True
