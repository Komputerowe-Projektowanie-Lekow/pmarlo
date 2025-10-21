from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pmarlo.utils.path_utils import ensure_directory

from .apply import apply_transform_plan
from .plan import TransformPlan, TransformStep
from .plan import to_text as plan_to_text
from .progress import ProgressCB, ProgressReporter

logger = logging.getLogger(__name__)


class TransformManifest:
    """Manages checkpoint manifest for transform runs."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.manifest_file = self.checkpoint_dir / ".pmarlo_transform_run.json"
        self.data: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """Load manifest from file."""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r") as f:
                    self.data = json.load(f)
                logger.info(f"Loaded transform manifest from {self.manifest_file}")
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
                self.data = {}
        return self.data

    def save(self):
        """Save manifest to file."""
        ensure_directory(self.checkpoint_dir)
        try:
            with open(self.manifest_file, "w") as f:
                json.dump(self.data, f, indent=2, default=str)
            logger.debug(f"Saved transform manifest to {self.manifest_file}")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    def init_run(self, plan: TransformPlan, run_id: Optional[str] = None):
        """Initialize a new run in the manifest."""
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        plan_hash = self._hash_plan(plan)
        self.data = {
            "run_id": run_id,
            "plan_hash": plan_hash,
            "step_index": 0,
            "step_name": None,
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "artifacts": {},
            "completed_steps": [],
            "failed_steps": [],
        }
        self.save()

    def _hash_plan(self, plan: TransformPlan) -> str:
        """Generate a hash for the plan to detect changes."""
        plan_str = json.dumps(
            [{"name": s.name, "params": s.params} for s in plan.steps], sort_keys=True
        )
        return hashlib.sha256(plan_str.encode()).hexdigest()[:16]

    def mark_step_start(self, step_index: int, step_name: str, params: Dict[str, Any]):
        """Mark a step as started."""
        self.data["step_index"] = step_index
        self.data["step_name"] = step_name
        self.data["params"] = params
        self.data["status"] = "running"
        self.save()

    def mark_step_complete(
        self,
        step_index: int,
        step_name: str,
        artifacts: Optional[Dict[str, Any]] = None,
    ):
        """Mark a step as completed."""
        completed_step = {
            "step_index": step_index,
            "step_name": step_name,
            "completed_at": datetime.now().isoformat(),
            "artifacts": artifacts or {},
        }

        # Remove from failed steps if it was there
        self.data["failed_steps"] = [
            s
            for s in self.data.get("failed_steps", [])
            if s.get("step_name") != step_name
        ]

        # Add to completed steps (or update if already there)
        completed_steps = self.data.get("completed_steps", [])
        self.data["completed_steps"] = [
            s for s in completed_steps if s.get("step_name") != step_name
        ]
        self.data["completed_steps"].append(completed_step)

        self.save()

    def mark_step_failed(self, step_index: int, step_name: str, error: str):
        """Mark a step as failed."""
        failed_step = {
            "step_index": step_index,
            "step_name": step_name,
            "failed_at": datetime.now().isoformat(),
            "error": error,
        }

        # Remove from completed if it was there
        self.data["completed_steps"] = [
            s
            for s in self.data.get("completed_steps", [])
            if s.get("step_name") != step_name
        ]

        # Add to failed steps (or update if already there)
        failed_steps = self.data.get("failed_steps", [])
        self.data["failed_steps"] = [
            s for s in failed_steps if s.get("step_name") != step_name
        ]
        self.data["failed_steps"].append(failed_step)

        self.data["status"] = "failed"
        self.save()

    def is_step_completed(self, step_name: str) -> bool:
        """Check if a step is already completed."""
        completed_steps = self.data.get("completed_steps", [])
        return any(s.get("step_name") == step_name for s in completed_steps)

    def get_last_completed_index(self) -> int:
        """Get the index of the last completed step."""
        completed_steps = self.data.get("completed_steps", [])
        if not completed_steps:
            return -1
        indices: list[int] = []
        for step in completed_steps:
            try:
                indices.append(int(step.get("step_index", -1)))
            except Exception:
                continue
        return max(indices) if indices else -1


def _initialize_manifest(
    plan: TransformPlan,
    checkpoint_dir: Optional[str | Path],
    run_id: Optional[str],
) -> Optional[TransformManifest]:
    if not checkpoint_dir:
        return None
    manifest = TransformManifest(Path(checkpoint_dir))
    manifest.load()
    if manifest.data and manifest.data.get("plan_hash") == manifest._hash_plan(plan):
        logger.info(f"Resuming existing run {manifest.data.get('run_id')}")
    else:
        manifest.init_run(plan, run_id)
    return manifest


def _should_skip_step(
    manifest: Optional[TransformManifest],
    idx: int,
    step_name: str,
    start_index: int,
) -> bool:
    if manifest is None:
        return False
    if idx < start_index:
        logger.info(f"Skipping completed step {idx}: {step_name}")
        return True
    if manifest.is_step_completed(step_name):
        logger.info(f"Step {step_name} already completed, skipping")
        return True
    return False


def _execute_step_with_retries(
    current_data: Any,
    step: TransformStep,
    idx: int,
    total_steps: int,
    *,
    manifest: Optional[TransformManifest],
    reporter: ProgressReporter,
    max_retries: int,
) -> Any:
    retry_count = 0
    while retry_count <= max_retries:
        try:
            t0 = time.time()
            reporter.emit(
                "aggregate_step_start",
                {"step_name": step.name, "index": idx + 1, "total_steps": total_steps},
            )
            if manifest:
                manifest.mark_step_start(idx, step.name, step.params)
            result = apply_transform_plan(current_data, TransformPlan(steps=(step,)))
            duration_s = round(time.time() - t0, 3)
            if manifest:
                manifest.mark_step_complete(idx, step.name, {"duration_s": duration_s})
            reporter.emit(
                "aggregate_step_end",
                {
                    "step_name": step.name,
                    "index": idx + 1,
                    "total_steps": total_steps,
                    "duration_s": duration_s,
                    "current_step": idx + 1,
                    "total_steps": total_steps,
                },
            )
            return result
        except Exception as exc:
            retry_count += 1
            logger.error(
                "Step %s failed (attempt %d/%d): %s",
                step.name,
                retry_count,
                max_retries + 1,
                exc,
            )
            if manifest:
                manifest.mark_step_failed(idx, step.name, str(exc))
            if retry_count > max_retries:
                reporter.emit("aggregate_end", {"status": "failed", "error": str(exc)})
                raise RuntimeError(
                    f"Step {step.name} failed after {max_retries} retries: {exc}"
                )
            time.sleep(min(2**retry_count, 30))


def apply_plan(
    plan: TransformPlan,
    data: Any,
    progress_callback: Optional[ProgressCB] = None,
    checkpoint_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    max_retries: int = 3,
) -> Any:
    """Apply a transform plan step-by-step with optional checkpointing.

    Args:
        plan: The transform plan to execute
        data: Input data to transform
        progress_callback: Optional progress callback
        checkpoint_dir: Directory for checkpoint files (enables checkpointing if provided)
        run_id: Optional run identifier
        max_retries: Maximum retries per step

    Events:
      - aggregate_begin: total_steps, plan_text
      - aggregate_step_start: step_name, index, total_steps
      - aggregate_step_end: step_name, index, total_steps, duration_s, current_step, total_steps
      - aggregate_end: status
    """
    reporter = ProgressReporter(progress_callback)
    steps: list[TransformStep] = list(plan.steps)

    manifest = _initialize_manifest(plan, checkpoint_dir, run_id)

    reporter.emit(
        "aggregate_begin",
        {"total_steps": len(steps), "plan_text": plan_to_text(plan)},
    )

    out = data
    total_steps = len(steps)
    start_index = manifest.get_last_completed_index() + 1 if manifest else 0

    for idx, step in enumerate(steps):
        if _should_skip_step(manifest, idx, step.name, start_index):
            continue
        out = _execute_step_with_retries(
            out,
            step,
            idx,
            total_steps,
            manifest=manifest,
            reporter=reporter,
            max_retries=max_retries,
        )

    if manifest:
        manifest.data["status"] = "completed"
        manifest.data["completed_at"] = datetime.now().isoformat()
        manifest.save()

    reporter.emit("aggregate_end", {"status": "ok"})
    return out
