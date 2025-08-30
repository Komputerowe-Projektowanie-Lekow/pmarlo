from __future__ import annotations

from pmarlo.transform.plan import TransformPlan, TransformStep
from pmarlo.transform.runner import apply_plan


def test_apply_plan_emits_aggregate_events():
    plan = TransformPlan(
        steps=(
            TransformStep("SMOOTH_FES", {"sigma": 0.5}),
            TransformStep("REORDER_STATES", {}),
        )
    )
    events: list[tuple[str, dict]] = []

    def cb(event: str, info):
        events.append((event, dict(info)))

    data = {"X": []}
    _ = apply_plan(plan, data, progress_callback=cb)

    names = [ev for ev, _ in events]
    assert names[0] == "aggregate_begin"
    assert names[-1] == "aggregate_end"

    # There should be start/end per step
    n_steps = 2
    assert names.count("aggregate_step_start") == n_steps
    assert names.count("aggregate_step_end") == n_steps

    # Each step_end carries progress fields
    for ev, info in events:
        if ev == "aggregate_step_end":
            assert (
                "current_step" in info
                and "total_steps" in info
                and "duration_s" in info
            )
