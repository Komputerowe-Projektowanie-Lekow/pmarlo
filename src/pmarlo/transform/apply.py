from .plan import TransformPlan


def smooth_fes(dataset, **kwargs):
    return dataset


def reorder_states(dataset, **kwargs):
    return dataset


def fill_gaps(dataset, **kwargs):
    return dataset


def apply_transform_plan(dataset, plan: TransformPlan):
    for step in plan.steps:
        if step.name == "SMOOTH_FES":
            dataset = smooth_fes(dataset, **step.params)
        elif step.name == "REORDER_STATES":
            dataset = reorder_states(dataset, **step.params)
        elif step.name == "FILL_GAPS":
            dataset = fill_gaps(dataset, **step.params)
        # ... add others ...
    return dataset
