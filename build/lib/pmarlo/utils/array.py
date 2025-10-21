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
        Required when ``parts`` is empty to explicitly define the shape of the
        returned array.
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

    if shape is None:
        raise ValueError("No arrays to concatenate and no shape provided")

    empty_shape = tuple(int(dim) for dim in shape)
    return np.zeros(empty_shape, dtype=dtype)
