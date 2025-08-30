from __future__ import annotations

import time
from typing import Any, Mapping

from pmarlo.progress import ProgressReporter


def test_progress_reporter_emits_elapsed_and_eta():
    events: list[tuple[str, Mapping[str, Any]]] = []

    def cb(event: str, info: Mapping[str, Any]) -> None:
        events.append((event, dict(info)))

    r = ProgressReporter(cb, min_interval_s=0.0)
    r.emit("setup", {"message": "init"})
    # Simulate equilibrate with known total
    total_eq = 5
    for k in range(total_eq):
        r.emit("equilibrate", {"current_step": k + 1, "total_steps": total_eq})
        time.sleep(0.001)
    # Simulate production with periodic exchange
    total_prod = 10
    for k in range(total_prod):
        if (k + 1) % 2 == 0:
            r.emit(
                "exchange",
                {"sweep_index": (k + 1) // 2, "n_replicas": 4, "acceptance_mean": 0.25},
            )
        r.emit("simulate", {"current_step": k + 1, "total_steps": total_prod})
        time.sleep(0.001)
    r.emit("finished", {"status": "ok"})

    assert events[0][0] == "setup"
    assert events[-1][0] == "finished"

    # elapsed_s monotonic and eta exists when progress present
    prev_elapsed = -1.0
    saw_eta = False
    for ev, info in events:
        assert "elapsed_s" in info
        assert info["elapsed_s"] >= prev_elapsed
        prev_elapsed = info["elapsed_s"]
        if ev in {"equilibrate", "simulate"}:
            assert "current_step" in info and "total_steps" in info
            if info.get("current_step") and info.get("total_steps"):
                assert "eta_s" in info
                saw_eta = True
    assert saw_eta
