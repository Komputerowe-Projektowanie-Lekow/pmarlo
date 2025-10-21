from __future__ import annotations

"""Shared metrics used across the joint REMD<->CV workflow."""

from dataclasses import dataclass, field
from typing import Dict

__all__ = ["Metrics", "GuardrailReport"]


@dataclass(frozen=True)
class Metrics:
    """Lightweight container for validation metrics collected per iteration."""

    vamp2_val: float
    its_val: float
    ck_error: float
    notes: str = ""


@dataclass(frozen=True)
class GuardrailReport:
    """Outcome of guardrail checks for the joint workflow end-to-end."""

    vamp2_trend_ok: bool
    its_plateau_ok: bool
    ck_threshold_ok: bool
    notes: Dict[str, str] = field(default_factory=dict)
