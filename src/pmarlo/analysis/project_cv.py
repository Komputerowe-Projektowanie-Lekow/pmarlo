"""Helpers for working with projected collective variables."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Tuple

import numpy as np
from numpy.typing import NDArray

from pmarlo.ml.deeptica.whitening import apply_output_transform

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
        array (or the original values when metadata is missing) and ``applied``
        indicates whether whitening metadata was available.
    """

    arr = np.asarray(values, dtype=np.float64)
    if metadata is None:
        return arr, False

    getter = getattr(metadata, "get", None)
    if getter is None:
        return arr, False

    mean = getter("output_mean")
    transform = getter("output_transform")
    already_flag = getter("output_transform_applied")

    applied = bool(mean is not None and transform is not None)
    try:
        whitened = apply_output_transform(arr, mean, transform, already_flag)
    except Exception:
        return arr, False

    if applied:
        try:
            metadata["output_transform_applied"] = True  # type: ignore[index]
        except Exception:
            pass
    return np.asarray(whitened, dtype=np.float64), applied
