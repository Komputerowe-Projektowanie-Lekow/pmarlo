from .apply import apply_transform_plan
from .build import (
    AppliedOpts,
    BuildOpts,
    BuildResult,
    RunMetadata,
    build_result,
    default_fes_builder,
    default_tram_builder,
)

# Pipeline imports removed to avoid circular dependency
# Use: from pmarlo.transform.pipeline import Pipeline, run_pmarlo
from .plan import TransformPlan, TransformStep
from .planner import get_transform_plan
from .progress import (
    ProgressPrinter,
    ProgressReporter,
    coerce_progress_callback,
    console_progress_cb,
    tee_progress,
)
from .runner import apply_plan


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
    "apply_plan",
    "pm_get_plan",
    "pm_apply_plan",
    # Progress API
    "ProgressPrinter",
    "ProgressReporter",
    "console_progress_cb",
    "coerce_progress_callback",
    "tee_progress",
    # Build API
    "BuildOpts",
    "AppliedOpts",
    "RunMetadata",
    "BuildResult",
    "build_result",
    "default_fes_builder",
    "default_tram_builder",
]
