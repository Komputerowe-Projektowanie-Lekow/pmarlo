from __future__ import annotations

import sys
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


def console_progress_cb(  # noqa: C901
    prefix: str = "[pmarlo]", stream=None, min_interval_s: float = 0.5
) -> ProgressCB:  # noqa: C901
    """Return a progress callback that prints concise, human‑readable lines.

    Designed for console/terminal visibility (e.g., when running Streamlit).
    """

    stream = sys.stdout if stream is None else stream
    last = {"t": 0.0}

    def _fmt(ev: str, d: Mapping[str, Any]) -> str:
        msg = ev
        if ev in {"setup"}:
            msg = f"setup: {d.get('message', '...')}"
        elif ev in {"equilibrate", "simulate"}:
            cur = int(d.get("current_step", d.get("current", 0)))
            tot = int(d.get("total_steps", d.get("total", 0)))
            eta = d.get("eta_s", "?")
            msg = f"{ev}: {cur:>6d}/{tot:<6d} ETA {eta}s"
        elif ev == "exchange":
            acc = round(float(d.get("acceptance_mean", 0)) * 100, 1)
            msg = f"exchange: sweep {d.get('sweep_index', '?')} acc {acc}%"
        elif ev.startswith("demux_"):
            if ev == "demux_begin":
                msg = f"demux: begin segments={d.get('segments', '?')}"
            elif ev == "demux_segment":
                msg = f"demux: segment {d.get('index', '?')}"
            elif ev == "demux_gap_fill":
                msg = f"demux: gap+{d.get('frames', '?')}"
            elif ev == "demux_end":
                msg = f"demux: end frames={d.get('frames', '?')} file={d.get('file', '?')}"
        elif ev.startswith("emit_"):
            if ev == "emit_begin":
                msg = (
                    f"emit: inputs={d.get('n_inputs', '?')} out={d.get('out_dir', '?')}"
                )
            elif ev == "emit_one_begin":
                msg = f"emit: start {d.get('traj', '?')}"
            elif ev == "emit_one_end":
                msg = f"emit: done {d.get('traj', '?')} -> {d.get('shard', '?')}"
            elif ev == "emit_end":
                msg = f"emit: n_shards={d.get('n_shards', '?')}"
        elif ev.startswith("aggregate_"):
            if ev == "aggregate_begin":
                msg = (
                    f"build: begin steps={d.get('total_steps', '?')} "
                    f"plan={d.get('plan_text', '')}"
                )
            elif ev == "aggregate_step_start":
                msg = (
                    f"build: step {d.get('index', '?')}/{d.get('total_steps', '?')} "
                    f"{d.get('step_name', '?')}"
                )
            elif ev == "aggregate_step_end":
                msg = (
                    f"build: step end {d.get('step_name', '?')} "
                    f"{d.get('duration_s', '?')}s"
                )
            elif ev == "aggregate_end":
                msg = f"build: end status={d.get('status', '?')}"
        elif ev == "finished":
            msg = f"finished: {d.get('status', 'ok')}"
        return f"{prefix} {msg}"

    def _cb(event: str, payload: Mapping[str, Any]) -> None:
        from time import time as _now

        t = _now()
        if (t - last["t"]) < float(min_interval_s) and not _is_boundary_event(
            event, payload
        ):
            return
        last["t"] = t
        try:
            print(_fmt(event, payload), file=stream, flush=True)
        except Exception:
            pass

    return _cb


def tee_progress(*callbacks: Optional[ProgressCB]) -> ProgressCB:
    """Combine multiple progress callbacks into one.

    Accepts a variable number of optional callbacks and returns a single
    callback that forwards events to each non-None callback.
    """
    cbs: list[ProgressCB] = [cb for cb in callbacks if cb is not None]

    def _cb(event: str, payload: Mapping[str, Any]) -> None:
        for cb in cbs:
            try:
                cb(event, payload)
            except Exception:
                pass

    return _cb


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
