from __future__ import annotations

import pytest

from pmarlo.transform.apply import apply_transform_plan
from pmarlo.transform.plan import TransformPlan, TransformStep


def test_apply_transform_plan_unknown_step_raises():
    plan = TransformPlan(steps=(TransformStep(name="NOT_A_STEP", params={}),))

    with pytest.raises(KeyError, match="Unknown transform step 'NOT_A_STEP'"):
        apply_transform_plan({}, plan)
