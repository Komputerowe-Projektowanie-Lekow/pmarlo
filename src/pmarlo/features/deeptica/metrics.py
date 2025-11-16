from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

__all__ = ["normalize_training_metrics"]


def _coerce_finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field} must be a numeric value") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite (received {value!r})")
    return result


def _coerce_positive_int(value: Any, *, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{field} must be positive (received {value!r})")
    return result


def normalize_training_metrics(
    metrics: Mapping[str, Any] | None,
    *,
    tau_schedule: Optional[Sequence[Any]] = None,
    epochs_per_tau: Optional[int | float] = None,
) -> Dict[str, Any]:
    """Validate training metrics now that training backends provide best scalars.

    ``tau_schedule`` and ``epochs_per_tau`` remain in the signature for API
    compatibility but no longer influence the returned values.
    """

    del tau_schedule, epochs_per_tau  # Preserve signature while silencing lint

    if not isinstance(metrics, Mapping):
        raise TypeError("normalize_training_metrics expects a mapping of metrics")

    normalized: Dict[str, Any] = dict(metrics)
    required = ("best_val_score", "best_epoch", "best_tau")
    missing = [key for key in required if normalized.get(key) is None]
    if missing:
        raise ValueError(
            f"training metrics missing required fields: {', '.join(sorted(missing))}"
        )

    normalized["best_val_score"] = _coerce_finite_float(
        normalized["best_val_score"], field="best_val_score"
    )
    normalized["best_epoch"] = _coerce_positive_int(
        normalized["best_epoch"], field="best_epoch"
    )
    normalized["best_tau"] = _coerce_positive_int(
        normalized["best_tau"], field="best_tau"
    )

    return normalized
