import json
from dataclasses import asdict, dataclass
from typing import Any, Literal

TransformName = Literal[
    "LEARN_CV",
    "SMOOTH_FES",
    "MERGE_BINS",
    "FILL_GAPS",
    "GROUP_TOP",
    "REORDER_STATES",
    "COARSE_GRAIN_MSM",
    "CLIP_OUTLIERS",
    # Terminal/estimator markers (for UI completeness)
    "MSM",
    "FES",
    "TRAM",
]


@dataclass(frozen=True)
class TransformStep:
    name: TransformName
    params: dict[str, Any]


@dataclass(frozen=True)
class TransformPlan:
    steps: tuple[TransformStep, ...]


def to_json(plan: TransformPlan) -> str:
    return json.dumps(
        {"version": 1, "steps": [asdict(s) for s in plan.steps]},
        separators=(",", ":"),
        ensure_ascii=False,
    )


def from_json(text: str) -> TransformPlan:
    obj = json.loads(text)
    steps = tuple(
        TransformStep(name=st["name"], params=dict(st.get("params", {})))
        for st in obj.get("steps", [])
    )
    return TransformPlan(steps=steps)


def to_dict(plan: TransformPlan) -> dict:
    """Return a structured dict representation suitable for UIs."""
    return {"version": 1, "steps": [asdict(s) for s in plan.steps]}


def to_text(plan: TransformPlan) -> str:
    return " → ".join(
        [
            s.name if not s.params else f"{s.name}({short_params(s.params)})"
            for s in plan.steps
        ]
    )


def short_params(d: dict[str, Any]) -> str:
    items: list[str] = []
    for k, v in list(d.items())[:4]:
        items.append(f"{k}={v}")
    if len(d) > 4:
        items.append("…")
    return ",".join(items)
