"""Legacy helpers retained for backwards compatibility."""

from __future__ import annotations

import warnings
from typing import Any

__all__ = [
    "prepare_batch",
    "compute_loss_and_score",
    "compute_grad_norm",
    "make_metrics",
    "record_metrics",
    "checkpoint_if_better",
]

warnings.warn(
    "pmarlo.features.deeptica_trainer.loops is deprecated; use "
    "pmarlo.ml.deeptica.trainer instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:  # pragma: no cover - optional ML stack
    from pmarlo.ml.deeptica.trainer import (  # type: ignore
        checkpoint_if_better,
        compute_grad_norm,
        compute_loss_and_score,
        make_metrics,
        prepare_batch,
        record_metrics,
    )
except Exception:  # pragma: no cover - torch/ML optional dependency

    def _missing(*_args: Any, **_kwargs: Any) -> Any:
        raise NotImplementedError(
            "DeepTICA training helpers require the optional 'pmarlo[mlcv]' extras"
        )

    prepare_batch = _missing  # type: ignore[assignment]
    compute_loss_and_score = _missing  # type: ignore[assignment]
    compute_grad_norm = _missing  # type: ignore[assignment]
    make_metrics = _missing  # type: ignore[assignment]
    record_metrics = _missing  # type: ignore[assignment]
    checkpoint_if_better = _missing  # type: ignore[assignment]
