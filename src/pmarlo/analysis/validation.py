"""Feature matrix validation utilities executed prior to MSM clustering."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from pmarlo import constants as const

__all__ = ["ValidationError", "validate_features"]

logger = logging.getLogger("pmarlo")


class ValidationError(RuntimeError):
    """Raised when continuous CV features fail numerical sanity checks."""

    def __init__(self, message: str, *, code: str, stats: Dict[str, Any]) -> None:
        self.code = str(code)
        self.stats = dict(stats)
        summary = json.dumps(self.stats, sort_keys=True, default=_json_default)
        super().__init__(f"{message} [code={self.code}] stats={summary}")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _normalise_feature_names(names: Sequence[str] | None, n_features: int) -> List[str]:
    if names is None:
        return [f"feature_{idx}" for idx in range(n_features)]
    cleaned: List[str] = [str(name) for name in names][:n_features]
    if len(cleaned) < n_features:
        cleaned.extend(f"feature_{idx}" for idx in range(len(cleaned), n_features))
    return cleaned


def _collect_column_stats(
    array: np.ndarray, names: Iterable[str]
) -> Dict[str, List[float]]:
    means: List[float] = []
    stds: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []

    for idx, name in enumerate(names):
        column = array[:, idx]
        finite_mask = np.isfinite(column)
        finite_vals = column[finite_mask]
        if finite_vals.size == 0:
            mean = float("nan")
            std = float("nan")
            min_val = float("nan")
            max_val = float("nan")
        else:
            mean = float(np.mean(finite_vals))
            std = float(np.std(finite_vals, ddof=0))
            min_val = float(np.min(finite_vals))
            max_val = float(np.max(finite_vals))
        means.append(mean)
        stds.append(std)
        mins.append(min_val)
        maxs.append(max_val)

        logger.info(
            "CV validation stats %s: mean=%.6g std=%.6g min=%.6g max=%.6g",
            name,
            mean,
            std,
            min_val,
            max_val,
        )

    return {
        "means": means,
        "stds": stds,
        "mins": mins,
        "maxs": maxs,
    }


def validate_features(
    X: np.ndarray | Sequence[Sequence[float]],
    feature_names: Sequence[str] | None,
) -> Dict[str, Any]:
    """Validate that CV features are finite and exhibit non-zero variance.

    Parameters
    ----------
    X
        2D array-like structure containing continuous collective variables.
    feature_names
        Optional sequence of feature names for logging and reporting; when
        omitted or size-mismatched, generic ``feature_{i}`` labels are used.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing column-wise statistics and finite row counts.

    Raises
    ------
    ValidationError
        When no rows contain exclusively finite values (``code=cv_no_finite_rows``),
        any element is NaN/Inf (``code=cv_non_finite``), or a column exhibits
        zero / numerically negligible standard deviation (``code=cv_zero_std``).
    """

    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {array.shape}")

    n_rows, n_features = array.shape
    names = _normalise_feature_names(feature_names, n_features)

    finite_matrix = np.isfinite(array)
    finite_rows_mask = finite_matrix.all(axis=1)
    finite_rows = int(finite_rows_mask.sum())
    non_finite_entries = int(finite_matrix.size - int(np.count_nonzero(finite_matrix)))

    logger.info(
        "CV validation: %d/%d rows contain only finite values", finite_rows, n_rows
    )

    column_stats = _collect_column_stats(array, names)

    stats: Dict[str, Any] = {
        "feature_names": names,
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "finite_rows": finite_rows,
        "non_finite_entries": non_finite_entries,
    }
    stats.update(column_stats)

    if finite_rows == 0:
        raise ValidationError(
            "No rows with fully finite CV values detected",
            code="cv_no_finite_rows",
            stats=stats,
        )

    if non_finite_entries > 0:
        raise ValidationError(
            "CV matrix contains non-finite values",
            code="cv_non_finite",
            stats=stats,
        )

    problematic: List[str] = []
    stds = column_stats["stds"]
    for name, std in zip(names, stds):
        if not np.isfinite(std) or std <= const.NUMERIC_MIN_POSITIVE:
            problematic.append(name)

    if problematic:
        extra = {"problematic_features": problematic}
        extra.update(stats)
        raise ValidationError(
            "Detected CV columns with zero or invalid standard deviation",
            code="cv_zero_std",
            stats=extra,
        )

    return stats
