from .apply import apply_transform_plan
from .plan import TransformPlan, TransformStep
from .planner import get_transform_plan


def pm_get_plan(dataset):
    plan = get_transform_plan(dataset)
    setattr(dataset, "transform_plan", plan)
    return dataset


def pm_apply_plan(dataset):
    plan = getattr(dataset, "transform_plan", None)
    if plan is None:
        return dataset
    return apply_transform_plan(dataset, plan)


__all__ = [
    "TransformPlan",
    "TransformStep",
    "get_transform_plan",
    "apply_transform_plan",
    "pm_get_plan",
    "pm_apply_plan",
]
