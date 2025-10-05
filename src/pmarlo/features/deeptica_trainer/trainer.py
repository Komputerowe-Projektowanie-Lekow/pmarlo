"""Thin facade over the canonical DeepTICA curriculum trainer."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from .config import TrainerConfig

try:  # pragma: no cover - optional ML stack
    from pmarlo.ml.deeptica.trainer import DeepTICACurriculumTrainer as _TrainerImpl
except Exception:  # pragma: no cover - torch/ML optional dependency
    _TrainerImpl = None  # type: ignore[assignment]


class _FallbackDeepTICATrainer:  # type: ignore[too-many-instance-attributes]
    """Fallback trainer stub when the ML stack is unavailable."""

    def __init__(
        self, model: Any, cfg: TrainerConfig, *args: Any, **kwargs: Any
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.args = args
        self.kwargs = kwargs

    def step(self, _batch: Iterable[Any]) -> None:
        raise NotImplementedError("DeepTICATrainer requires the optional mlcv extras")

    def evaluate(self, _batch: Iterable[Any]) -> None:
        raise NotImplementedError("DeepTICATrainer requires the optional mlcv extras")

    def fit(self, _train: Sequence[Any], *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("DeepTICATrainer requires the optional mlcv extras")


DeepTICATrainer: type[Any]
if _TrainerImpl is not None:
    DeepTICATrainer = _TrainerImpl
else:
    DeepTICATrainer = _FallbackDeepTICATrainer

__all__ = ["TrainerConfig", "DeepTICATrainer"]
