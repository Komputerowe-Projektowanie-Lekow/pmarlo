from __future__ import annotations

import time
from typing import Any, Optional

from ..progress import ProgressCB, ProgressReporter
from .apply import apply_transform_plan
from .plan import TransformPlan, TransformStep
from .plan import to_text as plan_to_text


def apply_plan(
    plan: TransformPlan,
    data: Any,
    progress_callback: Optional[ProgressCB] = None,
) -> Any:
    """Apply a transform plan step-by-step while emitting aggregate_* events.

    Events:
      - aggregate_begin: total_steps, plan_text
      - aggregate_step_start: step_name, index, total_steps
      - aggregate_step_end: step_name, index, total_steps, duration_s, current_step, total_steps
      - aggregate_end: status
    """
    reporter = ProgressReporter(progress_callback)
    steps: list[TransformStep] = list(plan.steps)
    reporter.emit(
        "aggregate_begin",
        {"total_steps": len(steps), "plan_text": plan_to_text(plan)},
    )
    out = data
    n = len(steps)
    for idx, step in enumerate(steps, start=1):
        t0 = time.time()
        reporter.emit(
            "aggregate_step_start",
            {"step_name": step.name, "index": idx, "total_steps": n},
        )
        # Reuse existing dispatch by applying a single-step plan
        out = apply_transform_plan(out, TransformPlan(steps=(step,)))
        reporter.emit(
            "aggregate_step_end",
            {
                "step_name": step.name,
                "index": idx,
                "total_steps": n,
                "duration_s": round(time.time() - t0, 3),
                "current_step": idx,
                "total_steps": n,
            },
        )
    reporter.emit("aggregate_end", {"status": "ok"})
    return out
