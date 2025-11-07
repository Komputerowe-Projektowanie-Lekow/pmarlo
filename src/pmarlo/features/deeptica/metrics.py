from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

__all__ = ["normalize_training_metrics"]


def normalize_training_metrics(
    metrics: Mapping[str, Any] | None,
    *,
    tau_schedule: Optional[Sequence[Any]] = None,
    epochs_per_tau: Optional[int | float] = None,
) -> Dict[str, Any]:
    """Return training history metrics with best score/epoch/tau inferred.

    Parameters
    ----------
    metrics:
        Original metrics mapping collected during training.
    tau_schedule:
        Sequence of lag times that were traversed during curriculum training.
    epochs_per_tau:
        Number of epochs spent at each tau stage. Required to infer ``best_tau``
        when the training metadata does not contain it explicitly.
    """

    if not isinstance(metrics, Mapping):
        return {}

    normalized: MutableMapping[str, Any] = dict(metrics)

    raw_curve = normalized.get("val_score_curve")
    finite_scores: List[tuple[int, float]] = []
    if isinstance(raw_curve, Sequence):
        for idx, value in enumerate(raw_curve):
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                finite_scores.append((idx, score))

    best_val_score = normalized.get("best_val_score")
    if best_val_score is None and finite_scores:
        best_idx, best_score = max(finite_scores, key=lambda item: item[1])
        normalized["best_val_score"] = float(best_score)
        normalized.setdefault("_best_epoch_index", best_idx)
    elif finite_scores and isinstance(normalized.get("best_epoch"), (int, float)):
        idx = int(normalized["best_epoch"]) - 1
        if 0 <= idx < len(finite_scores):
            normalized.setdefault("_best_epoch_index", idx)

    best_epoch = normalized.get("best_epoch")
    if best_epoch is None and finite_scores:
        best_idx = normalized.get("_best_epoch_index")
        if not isinstance(best_idx, int):
            best_idx = max(finite_scores, key=lambda item: item[1])[0]
        normalized["best_epoch"] = int(best_idx + 1)
        if normalized.get("best_val_score") is None:
            normalized["best_val_score"] = float(finite_scores[best_idx][1])
    elif isinstance(best_epoch, (int, float)):
        normalized["best_epoch"] = int(best_epoch)

    if normalized.get("best_val_score") is not None:
        try:
            normalized["best_val_score"] = float(normalized["best_val_score"])
        except (TypeError, ValueError):
            normalized["best_val_score"] = None

    best_tau = normalized.get("best_tau")
    if best_tau is None:
        schedule: List[int] = []
        if isinstance(tau_schedule, Sequence):
            for item in tau_schedule:
                try:
                    schedule.append(int(item))
                except (TypeError, ValueError):
                    continue
        epochs = None
        if isinstance(epochs_per_tau, (int, float)):
            epochs = int(epochs_per_tau)
        if schedule and epochs and epochs > 0:
            idx = normalized.get("_best_epoch_index")
            if not isinstance(idx, int):
                if finite_scores:
                    idx = max(finite_scores, key=lambda item: item[1])[0]
                else:
                    idx = None
            if isinstance(idx, int):
                stage = max(0, min(idx // epochs, len(schedule) - 1))
                normalized["best_tau"] = schedule[stage]
    else:
        try:
            normalized["best_tau"] = int(best_tau)
        except (TypeError, ValueError):
            normalized["best_tau"] = None

    normalized.pop("_best_epoch_index", None)
    return dict(normalized)
