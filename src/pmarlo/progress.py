from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Optional, Tuple

ProgressCB = Callable[[str, Mapping[str, Any]], None]

# Accepted kwarg aliases for progress callbacks in public APIs
ALIAS_KEYS = (
    "progress_callback",
    "callback",
    "on_event",
    "progress",
    "reporter",
)


def coerce_progress_callback(kwargs: dict) -> Optional[ProgressCB]:
    """Extract a callback from common alias kwargs and normalize the key.

    Leaves the original kwargs untouched except for setting
    ``progress_callback`` when any alias was provided.
    """
    cb = None
    for k in ALIAS_KEYS:
        if k in kwargs and kwargs[k] is not None:
            cb = kwargs[k]
            break
    if cb is not None:
        # Normalize for downstream code if they inspect kwargs
        kwargs.setdefault("progress_callback", cb)
    return cb


class ProgressReporter:
    """Lightweight event emitter with elapsed/ETA and rate limiting.

    - Adds wall-clock ``elapsed_s`` on every event.
    - Computes ``eta_s`` when given (current_step,total_steps) or (current,total).
    - Rate-limits high-frequency duplicates via ``min_interval_s``.
    - Never raises if the user callback fails.
    """

    def __init__(self, cb: Optional[ProgressCB], min_interval_s: float = 0.4):
        self._cb = cb
        self._t0 = time.time()
        self._last_emit = 0.0
        self._min_interval = float(min_interval_s)

    def emit(self, event: str, info: Optional[Mapping[str, Any]] = None) -> None:
        cb = self._cb
        if cb is None:
            return
        now = time.time()
        if (now - self._last_emit) < self._min_interval and not _is_boundary_event(
            event, info
        ):
            return  # Drop frequent duplicates
        self._last_emit = now

        payload = dict(info or {})
        payload["elapsed_s"] = round(now - self._t0, 3)

        cur, tot = _extract_progress(payload)
        if cur is not None and tot and tot > 0 and cur <= tot:
            frac = max(1e-9, min(1.0, cur / float(tot)))
            eta = (now - self._t0) * (1.0 / frac - 1.0)
            payload["eta_s"] = round(eta, 3)

        try:
            cb(event, payload)
        except Exception:
            # Never propagate UI/reporting errors to the core algorithms
            pass


def _extract_progress(d: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    # accept current/total or current_step/total_steps
    cur = d.get("current", d.get("current_step"))
    tot = d.get("total", d.get("total_steps"))
    try:
        return (
            (float(cur), float(tot))
            if cur is not None and tot is not None
            else (None, None)
        )
    except Exception:
        return (None, None)


def _is_boundary_event(event: str, info: Optional[Mapping[str, Any]]) -> bool:
    # Events we do not rate-limit because they delineate stages
    return event in {
        "setup",
        "equilibrate",
        "simulate",
        "exchange",
        "write_output",
        "finished",
        "aggregate_begin",
        "aggregate_step_start",
        "aggregate_step_end",
        "aggregate_end",
    }
