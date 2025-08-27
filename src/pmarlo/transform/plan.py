from dataclasses import dataclass
from typing import Any, Literal

TransformName = Literal[
    "SMOOTH_FES", "MERGE_BINS", "FILL_GAPS", "GROUP_TOP",
    "REORDER_STATES", "COARSE_GRAIN_MSM", "CLIP_OUTLIERS"
]

@dataclass(frozen=True)
class TransformStep:
    name: TransformName
    params: dict[str, Any]

@dataclass(frozen=True)
class TransformPlan:
    steps: tuple[TransformStep, ...]
