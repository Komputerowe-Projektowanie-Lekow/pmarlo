import os

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def get_bool_env(name: str, *, default: bool = False) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized_value = raw_value.strip().lower()

    if normalized_value in _TRUE_VALUES:
        return True

    if normalized_value in _FALSE_VALUES:
        return False

    raise ValueError(
        f"Invalid boolean value for environment variable {name!r}: " f"{raw_value!r}."
    )


FES_SMOOTHING = get_bool_env("PMARLO_FES_SMOOTHING", default=False)
REORDER_STATES = get_bool_env("PMARLO_REORDER_STATES", default=False)
JOINT_USE_REWEIGHT = get_bool_env("PMARLO_JOINT_USE_REWEIGHT", default=True)
