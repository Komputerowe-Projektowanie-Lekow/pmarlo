from ..config import FES_SMOOTHING, REORDER_STATES
from .plan import TransformPlan, TransformStep


def get_transform_plan(dataset) -> TransformPlan:
    steps: list[TransformStep] = []

    # canonical metadata only
    shape_any = getattr(dataset, "output_shape", None)

    # heuristics – use canonical shape, not ad-hoc internals
    if FES_SMOOTHING.get():
        steps.append(TransformStep("SMOOTH_FES", {"method": "gaussian", "sigma": 1.0}))

    if (
        REORDER_STATES.get()
        and isinstance(shape_any, tuple)
        and len(shape_any) > 0
        and int(shape_any[0]) >= 128
    ):
        steps.append(TransformStep("REORDER_STATES", {"criterion": "degree"}))

    # demux gap repair only if allowed and gaps detected
    if (
        getattr(dataset, "has_gaps", False)
        and getattr(dataset, "metadata", None)
        and getattr(dataset.metadata, "allow_repair", False)
    ):
        steps.append(TransformStep("FILL_GAPS", {"strategy": "nearest"}))

    return TransformPlan(tuple(steps))
