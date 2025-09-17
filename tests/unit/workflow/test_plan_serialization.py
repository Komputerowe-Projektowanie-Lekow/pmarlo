from __future__ import annotations

from pmarlo.transform.plan import (
    TransformPlan,
    TransformStep,
    from_json,
    to_json,
    to_text,
)


def test_plan_json_roundtrip_and_text():
    plan = TransformPlan(
        steps=(
            TransformStep("LEARN_CV", {"method": "deeptica"}),
            TransformStep("SMOOTH_FES", {"sigma": 0.5, "window": 5}),
        )
    )
    s = to_json(plan)
    plan2 = from_json(s)
    assert tuple(plan2.steps) == tuple(plan.steps)

    txt = to_text(plan)
    assert isinstance(txt, str) and len(txt) > 0 and "SMOOTH_FES" in txt
