"""Configuration helpers for DeepTICA workflows."""

from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping

from pmarlo.utils.json_io import sanitize

__all__ = ["resolve_deeptica", "sanitize_deeptica_payload"]


def resolve_deeptica(
    transform_cfg: Mapping[str, Any],
) -> tuple[bool, Dict[str, Any] | None]:
    """Parse and validate DeepTICA configuration sections."""

    deeptica_cfg = transform_cfg.get("deeptica")
    if not isinstance(deeptica_cfg, MutableMapping):
        return False, None
    cfg = dict(deeptica_cfg)
    enabled = bool(cfg.pop("enabled", True))

    if "skip_on_failure" in cfg:
        cfg["skip_on_failure"] = bool(cfg["skip_on_failure"])

    if "min_pairs" in cfg:
        try:
            cfg["min_pairs"] = int(cfg["min_pairs"])
        except (TypeError, ValueError):
            cfg.pop("min_pairs", None)

    return enabled, cfg if cfg else None


def sanitize_deeptica_payload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the stable summary fields from DeepTICA training results."""

    summary: Dict[str, Any] = {}
    fields = [
        "applied",
        "skipped",
        "reason",
        "method",
        "lag",
        "lag_used",
        "n_out",
        "pairs_total",
        "warnings",
        "lag_candidates",
    ]
    for field in fields:
        if field in raw:
            summary[field] = raw[field]

    attempts = raw.get("attempts")
    if isinstance(attempts, list):
        trimmed: list[Dict[str, Any]] = []
        for attempt in attempts[:5]:
            if not isinstance(attempt, Mapping):
                continue
            trimmed.append(
                {
                    "lag": attempt.get("lag"),
                    "pairs_total": attempt.get("pairs_total"),
                    "status": attempt.get("status"),
                    "warnings": attempt.get("warnings"),
                }
            )
        if trimmed:
            summary["attempts"] = trimmed

    training_metrics = raw.get("training_metrics")
    if isinstance(training_metrics, Mapping):
        summary["training_metrics"] = {
            "wall_time_s": training_metrics.get("wall_time_s"),
            "final_objective": training_metrics.get("final_objective"),
            "output_variance": training_metrics.get("output_variance"),
        }
    return sanitize(summary)
