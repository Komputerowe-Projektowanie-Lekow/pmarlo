"""Compatibility façade around the canonical DeepTICA curriculum config."""

from __future__ import annotations

from typing import Any, Iterable, List

from pmarlo.ml.deeptica.trainer import CurriculumConfig

__all__ = ["TrainerConfig", "resolve_curriculum"]


def _coerce_schedule(spec: Any) -> tuple[int, ...]:
    if spec is None:
        return ()
    if isinstance(spec, Iterable) and not isinstance(spec, (str, bytes)):
        values = [int(x) for x in spec if int(x) > 0]
    else:
        step = int(spec)
        values = [step] if step > 0 else []
    return tuple(values)


class TrainerConfig(CurriculumConfig):  # type: ignore[misc]
    """Wrapper accepting the historical ``tau_steps`` argument."""

    def __init__(self, tau_steps: Any = None, **kwargs: Any) -> None:
        schedule_input = kwargs.pop("tau_schedule", None)
        schedule: tuple[int, ...] = ()
        if schedule_input is not None:
            schedule = _coerce_schedule(schedule_input)
        elif tau_steps is not None:
            schedule = _coerce_schedule(tau_steps)
        if schedule:
            kwargs["tau_schedule"] = schedule
        elif tau_steps is not None or schedule_input is not None:
            raise ValueError("tau steps must be positive")

        scheduler = kwargs.pop("scheduler", None)
        total_steps = kwargs.pop("total_steps", None)
        alt_total = kwargs.pop("scheduler_total_steps", None)
        if total_steps is None and alt_total is not None:
            total_steps = alt_total

        grad_clip_mode = kwargs.pop("grad_clip_mode", None)
        log_every = kwargs.pop("log_every", None)

        if scheduler == "cosine" and not total_steps:
            raise ValueError("cosine scheduler requires total_steps")

        super().__init__(**kwargs)

        if schedule:
            object.__setattr__(self, "tau_schedule", schedule)

        if scheduler is not None:
            object.__setattr__(self, "scheduler", scheduler)
        if total_steps is not None:
            object.__setattr__(self, "scheduler_total_steps", int(total_steps))
        if grad_clip_mode is not None:
            object.__setattr__(self, "grad_clip_mode", str(grad_clip_mode).lower())
        if log_every is not None:
            object.__setattr__(self, "log_every", max(1, int(log_every)))


def resolve_curriculum(cfg: TrainerConfig) -> List[int]:
    """Return the ordered tau curriculum as positive integers."""

    schedule: Iterable[int] = getattr(cfg, "tau_schedule", ())
    return [max(1, int(step)) for step in schedule]
