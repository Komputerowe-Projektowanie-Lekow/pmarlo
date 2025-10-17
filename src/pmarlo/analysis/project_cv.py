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
        array and ``applied`` is always ``True`` when metadata validation
        succeeds.
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

    whitened = apply_output_transform(arr, mean, transform, already_flag)

    if isinstance(metadata, MutableMapping):
        metadata["output_transform_applied"] = True  # type: ignore[index]

    return np.asarray(whitened, dtype=np.float64), True
