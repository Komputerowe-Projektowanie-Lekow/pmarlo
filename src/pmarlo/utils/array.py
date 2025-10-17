from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


def concatenate_or_empty(
    parts: Iterable[np.ndarray],
    *,
    dtype: np.dtype | type,
    shape: Sequence[int] | None = None,
    copy: bool = False,
) -> NDArray[np.generic]:
    """Concatenate array ``parts`` or return an empty array of ``dtype``.

    Parameters
    ----------
    parts:
        Iterable of array-like chunks to concatenate.
    dtype:
        Desired dtype of the resulting array.
    shape:
        Shape of the empty fallback array. Defaults to ``(0,)`` when not
        provided.
    copy:
        Passed to :meth:`numpy.ndarray.astype` when coercing the dtype.

    Returns
    -------
    numpy.ndarray
        Concatenated array when ``parts`` is non-empty, otherwise an empty
        array with the requested dtype and shape.
    """

    chunks = tuple(parts)
    if chunks:
        concatenated = np.concatenate(chunks)
        return concatenated.astype(dtype, copy=copy)

    fallback_shape = (0,) if shape is None else tuple(int(dim) for dim in shape)
    return np.zeros(fallback_shape, dtype=dtype)
