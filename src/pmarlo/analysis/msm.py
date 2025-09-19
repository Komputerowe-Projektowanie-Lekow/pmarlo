"""Utilities for preparing MSM inputs with learned CV whitening."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

import numpy as np

from .project_cv import apply_whitening_from_metadata


DatasetLike = MutableMapping[str, Any]


def ensure_msm_inputs_whitened(dataset: DatasetLike | Mapping[str, Any]) -> bool:
    """Ensure the continuous CV matrix in ``dataset`` is whitened.

    The function mutates ``dataset["X"]`` in-place when whitening metadata is
    available.  It returns ``True`` when metadata was found and applied.
    """

    if not isinstance(dataset, (MutableMapping, dict)):
        return False

    X = dataset.get("X")  # type: ignore[assignment]
    if X is None:
        return False

    artifacts = dataset.get("__artifacts__")  # type: ignore[assignment]
    summary: Any | None = None
    if isinstance(artifacts, Mapping):
        summary = artifacts.get("mlcv_deeptica")
    if not isinstance(summary, (MutableMapping, dict)):
        return False

    whitened, applied = apply_whitening_from_metadata(np.asarray(X, dtype=np.float64), summary)
    if applied:
        dataset["X"] = whitened  # type: ignore[index]
    return applied
