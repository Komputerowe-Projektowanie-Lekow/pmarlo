# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Checkpoint and Resume System for PMARLO
Handles state management for long-running molecular dynamics simulations.
"""

import json
import logging
import os
import pickle
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoints and resume functionality for all pipeline types."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        output_base_dir: str = "output",
        pipeline_steps: Optional[List[str]] = None,
        auto_continue: bool = False,
        max_retries: int = 3,
    ):
        """
        Initialize checkpoint manager.

        Args:
            run_id: Unique identifier for this run (5-digit string)
            output_base_dir: Base directory for all outputs
            pipeline_steps: Custom list of pipeline steps (uses default if None)
            auto_continue: Automatically detect and continue interrupted runs
            max_retries: Maximum retry attempts for a failed step
        """
        self.output_base_dir = Path(output_base_dir)
        self.run_id = run_id or self._generate_run_id()
        self.run_dir = self.output_base_dir / self.run_id
        self.life_file = self.run_dir / "life.json"
        self.state_file = self.run_dir / "state.pkl"
        self.config_file = self.run_dir / "config.json"
        self.auto_continue = auto_continue
        self.max_retries = max_retries

        # Track start times for running steps (not persisted)
        self.step_start_times: Dict[str, datetime] = {}

        # Ensure output directory exists
        self.output_base_dir.mkdir(exist_ok=True)

        # Default steps for REMD pipeline (backwards compatibility)
        default_steps = [
            "protein_preparation",
            "system_setup",
            "replica_initialization",
            "energy_minimization",
            "gradual_heating",
            "equilibration",
            "production_simulation",
            "trajectory_demux",
            "trajectory_analysis",
        ]

        # Handle auto-continue logic
        if auto_continue and self.life_file.exists():
            logger.info(f"Auto-continuing existing run {self.run_id}")
            self.load_life_data()
        else:
            # Initialize life tracking
            self.life_data: Dict[str, Any] = {
                "run_id": self.run_id,
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "current_stage": "initialization",
                "completed_steps": [],
                "failed_steps": [],
                "total_steps": pipeline_steps or default_steps,
                "status": "running",
                "pipeline_type": "remd" if pipeline_steps is None else "custom",
            }

        logger.info(f"Checkpoint Manager initialized for run {self.run_id}")

    def _generate_run_id(self) -> str:
        """Generate a unique 5-digit run ID."""
        import random
        import string

        return "".join(random.choices(string.digits, k=5))

    def setup_run_directory(self) -> Path:
        """Create and setup the run directory structure."""
        # Create run directory
        self.run_dir.mkdir(exist_ok=True)

        # Create subdirectories
        subdirs = ["trajectories", "analysis"]

        for subdir in subdirs:
            (self.run_dir / subdir).mkdir(exist_ok=True)

        # Save initial life file
        self.save_life_data()

        logger.info(f"Run directory setup complete: {self.run_dir}")
        return self.run_dir

    def save_life_data(self) -> None:
        """Save current life data to JSON file."""
        self.life_data["last_updated"] = datetime.now().isoformat()
        with open(self.life_file, "w") as f:
            json.dump(self.life_data, f, indent=2)

    def load_life_data(self) -> Dict[str, Any]:
        """Load life data from JSON file."""
        if self.life_file.exists():
            with open(self.life_file, "r") as f:
                self.life_data = json.load(f)
        return self.life_data

    def mark_step_started(self, step_name: str) -> None:
        """Mark a step as started and clear previous failures."""
        failed_steps = self.life_data["failed_steps"]
        if isinstance(failed_steps, list):
            for step in failed_steps:
                if isinstance(step, dict) and step.get("name") == step_name:
                    step["in_progress"] = True
                    break

        self.step_start_times[step_name] = datetime.now()
        self.life_data["current_stage"] = step_name
        self.life_data["status"] = "running"
        self.save_life_data()
        logger.info(f"Step started: {step_name}")

    def mark_step_completed(
        self, step_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark a step as completed."""
        step_data = {
            "name": step_name,
            "completed_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Remove from failed if it was there
        failed_steps = self.life_data["failed_steps"]
        if isinstance(failed_steps, list):
            self.life_data["failed_steps"] = [
                s
                for s in failed_steps
                if isinstance(s, dict) and s.get("name") != step_name
            ]

        # Add to completed (or update if already there)
        completed_steps = self.life_data["completed_steps"]
        if isinstance(completed_steps, list):
            self.life_data["completed_steps"] = [
                s
                for s in completed_steps
                if isinstance(s, dict) and s.get("name") != step_name
            ]
            self.life_data["completed_steps"].append(step_data)

        self.step_start_times.pop(step_name, None)
        self.save_life_data()
        logger.info(f"Step completed: {step_name}")

    def mark_step_failed(self, step_name: str, error_msg: str) -> None:
        """Mark a step as failed and handle retry tracking."""
        retries = 1

        # Remove from completed if it was there
        completed_steps = self.life_data["completed_steps"]
        if isinstance(completed_steps, list):
            self.life_data["completed_steps"] = [
                s
                for s in completed_steps
                if isinstance(s, dict) and s.get("name") != step_name
            ]

        failed_steps = self.life_data["failed_steps"]
        if not isinstance(failed_steps, list):
            failed_steps = []

        updated = False
        for step in failed_steps:
            if isinstance(step, dict) and step.get("name") == step_name:
                retries = int(step.get("retries", 0)) + 1
                step.update(
                    {
                        "failed_at": datetime.now().isoformat(),
                        "error": error_msg,
                        "retries": retries,
                        "in_progress": False,
                    }
                )
                updated = True
                break

        if not updated:
            failed_steps.append(
                {
                    "name": step_name,
                    "failed_at": datetime.now().isoformat(),
                    "error": error_msg,
                    "retries": retries,
                    "in_progress": False,
                }
            )

        self.life_data["failed_steps"] = failed_steps
        self.life_data["status"] = "failed"
        self.step_start_times.pop(step_name, None)

        if retries >= self.max_retries:
            completed_steps = self.life_data.get("completed_steps")
            if isinstance(completed_steps, list):
                completed_steps = [
                    s
                    for s in completed_steps
                    if isinstance(s, dict) and s.get("name") != step_name
                ]
                completed_steps.append(
                    {
                        "name": step_name,
                        "completed_at": datetime.now().isoformat(),
                        "metadata": {"failed": True, "error": error_msg},
                    }
                )
                self.life_data["completed_steps"] = completed_steps

            logger.error(
                f"Step failed: {step_name} - {error_msg} (max retries {self.max_retries} reached)"
            )
        else:
            logger.error(
                f"Step failed: {step_name} - {error_msg} (retry {retries}/{self.max_retries})"
            )

        self.save_life_data()

    def clear_failed_step(self, step_name: str) -> None:
        """Clear a step from failed list (when retrying)."""
        failed_steps = self.life_data["failed_steps"]
        if isinstance(failed_steps, list):
            self.life_data["failed_steps"] = [
                s
                for s in failed_steps
                if isinstance(s, dict) and s.get("name") != step_name
            ]

        # If no more failed steps, update status
        if not self.life_data["failed_steps"]:
            self.life_data["status"] = "running"

        self.step_start_times.pop(step_name, None)
        self.save_life_data()
        logger.info(f"Cleared failed status for step: {step_name}")

    def has_timed_out(self, step_name: str, timeout_seconds: int) -> bool:
        """Check if a running step has exceeded the timeout."""
        started_at = self.step_start_times.get(step_name)
        if not started_at:
            return False

        if datetime.now() - started_at > timedelta(seconds=timeout_seconds):
            logger.warning(
                f"Step {step_name} timed out after {timeout_seconds} seconds"
            )
            return True
        return False

    def is_step_completed(self, step_name: str) -> bool:
        """Check if a step has been completed."""
        completed_steps = self.life_data["completed_steps"]
        if isinstance(completed_steps, list):
            completed_names = [
                s.get("name") for s in completed_steps if isinstance(s, dict)
            ]
            return step_name in completed_names
        return False

    def get_next_step(self) -> Optional[str]:
        """Get the next step to execute.

        Delegates to smaller helpers to reduce complexity while preserving
        behavior.
        """
        completed_names = self._get_completed_step_names()
        failed_names = self._get_failed_step_names()

        # First priority: retry failed steps (most recent first)
        if self._has_failed_steps(failed_names):
            return self._select_failed_step_to_retry(failed_names)

        # Second priority: continue with next uncompleted step.
        # Find the next logical step after the last completed one.
        total_steps = self._get_total_steps_list()
        if total_steps is None:
            return None

        if self._has_completed_steps(completed_names):
            last_completed_idx = self._find_last_completed_index(
                total_steps, completed_names
            )
            return self._next_step_after_index(total_steps, last_completed_idx)

        # No steps completed yet, start from the beginning
        return self._first_step_if_any(total_steps)

    # --- Helper methods for get_next_step ---

    def _extract_names_from_step_dicts(self, steps: Any) -> List[str]:
        """Extract valid step names from a list of step dicts."""
        names: List[str] = []
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    name = step.get("name")
                    if isinstance(name, str):
                        names.append(name)
        return names

    def _get_completed_step_names(self) -> List[str]:
        """Return names of completed steps from life_data."""
        return self._extract_names_from_step_dicts(self.life_data["completed_steps"])

    def _get_failed_step_names(self) -> List[str]:
        """Return names of failed steps that can still be retried."""
        names: List[str] = []
        failed_steps = self.life_data["failed_steps"]
        if isinstance(failed_steps, list):
            for step in failed_steps:
                if isinstance(step, dict):
                    name = step.get("name")
                    retries = int(step.get("retries", 0))
                    in_progress = step.get("in_progress", False)
                    if (
                        isinstance(name, str)
                        and retries < self.max_retries
                        and not in_progress
                    ):
                        names.append(name)
        return names

    def _has_failed_steps(self, failed_names: List[str]) -> bool:
        """True if there is at least one failed step name."""
        return bool(failed_names)

    def _select_failed_step_to_retry(self, failed_names: List[str]) -> Optional[str]:
        """Select the most recent failed step to retry."""
        return failed_names[-1] if failed_names else None

    def _get_total_steps_list(self) -> Optional[List[str]]:
        """Return the configured total steps list, or None if invalid."""
        total_steps = self.life_data.get("total_steps")
        return total_steps if isinstance(total_steps, list) else None

    def _has_completed_steps(self, completed_names: List[str]) -> bool:
        """True if there is at least one completed step name."""
        return bool(completed_names)

    def _find_last_completed_index(
        self, total_steps: List[Any], completed_names: List[str]
    ) -> int:
        """Return highest index in total_steps that appears in completed_names."""
        last_completed_idx = -1
        for index, step in enumerate(total_steps):
            if isinstance(step, str) and step in completed_names:
                last_completed_idx = max(last_completed_idx, index)
        return last_completed_idx

    def _next_step_after_index(
        self, total_steps: List[Any], last_completed_idx: int
    ) -> Optional[str]:
        """Return the step name immediately after last_completed_idx if valid."""
        next_index = last_completed_idx + 1
        if next_index < len(total_steps):
            next_step = total_steps[next_index]
            return next_step if isinstance(next_step, str) else None
        return None

    def _first_step_if_any(self, total_steps: List[Any]) -> Optional[str]:
        """Return the first step if the list is non-empty and valid."""
        if total_steps:
            first_step = total_steps[0]
            return first_step if isinstance(first_step, str) else None
        return None

    def save_state(self, state_data: Dict[str, Any]) -> None:
        """Save arbitrary state data to pickle file."""
        with open(self.state_file, "wb") as f:
            pickle.dump(state_data, f)
        logger.info("State data saved to checkpoint")

    def load_state(self) -> Dict[str, Any]:
        """Load state data from pickle file."""
        if self.state_file.exists():
            with open(self.state_file, "rb") as f:
                state_data = pickle.load(f)
                return state_data if isinstance(state_data, dict) else {}
        return {}

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration for this run."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)
        logger.info("Configuration saved")

    def load_config(self) -> Dict[str, Any]:
        """Load configuration for this run."""
        if self.config_file.exists():
            with open(self.config_file, "r") as f:
                config_data = json.load(f)
                return config_data if isinstance(config_data, dict) else {}
        return {}

    def copy_input_files(self, files_to_copy: List[str]) -> None:
        """Copy input files to run directory for reproducibility."""
        input_dir = self.run_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files_to_copy:
            if os.path.exists(file_path):
                dest = input_dir / os.path.basename(file_path)
                shutil.copy2(file_path, dest)
                logger.info(f"Copied input file: {file_path} -> {dest}")

    def get_run_summary(self) -> Dict[str, Any]:
        """Get a summary of the current run status."""
        total_steps = self.life_data.get("total_steps", [])
        completed_steps = self.life_data.get("completed_steps", [])
        failed_steps = self.life_data.get("failed_steps", [])

        total_count = len(total_steps) if isinstance(total_steps, list) else 0
        completed_count = (
            len(completed_steps) if isinstance(completed_steps, list) else 0
        )
        failed_count = len(failed_steps) if isinstance(failed_steps, list) else 0

        return {
            "run_id": self.run_id,
            "status": self.life_data.get("status", "unknown"),
            "current_stage": self.life_data.get("current_stage", "unknown"),
            "progress": f"{completed_count}/{total_count}",
            "progress_percent": (
                (completed_count / total_count) * 100 if total_count > 0 else 0
            ),
            "completed_steps": completed_count,
            "failed_steps": failed_count,
            "created": self.life_data.get("created", ""),
            "last_updated": self.life_data.get("last_updated", ""),
            "run_directory": str(self.run_dir),
        }

    def print_status(self, verbose: bool = True) -> Dict[str, Any]:
        """Print current run status."""
        summary = self.get_run_summary()

        if not verbose:
            return summary

        print(f"\n{'='*60}")
        print(f"CHECKPOINT STATUS - Run ID: {summary['run_id']}")
        print(f"{'='*60}")
        print(f"Status: {summary['status'].upper()}")
        print(f"Current Stage: {summary['current_stage']}")
        print(f"Progress: {summary['progress']} ({summary['progress_percent']:.1f}%)")
        print(f"Directory: {summary['run_directory']}")
        print(f"Last Updated: {summary['last_updated']}")

        completed_steps = self.life_data.get("completed_steps", [])
        if isinstance(completed_steps, list) and completed_steps:
            print("\nCompleted Steps:")
            for step in completed_steps:
                if isinstance(step, dict):
                    name = step.get("name", "unknown")
                    completed_at = step.get("completed_at", "unknown")
                    print(f"  ✓ {name} ({completed_at})")

        failed_steps = self.life_data.get("failed_steps", [])
        if isinstance(failed_steps, list) and failed_steps:
            print("\nFailed Steps:")
            for step in failed_steps:
                if isinstance(step, dict):
                    name = step.get("name", "unknown")
                    error = step.get("error", "unknown error")
                    print(f"  ✗ {name} - {error}")

        next_step = self.get_next_step()
        if next_step:
            print(f"\nNext Step: {next_step}")
        else:
            print("\nAll steps completed!")

        print(f"{'='*60}\n")
        return summary

    @staticmethod
    def find_existing_runs(output_base_dir: str = "output") -> List[str]:
        """Find all existing run IDs in the output directory."""
        output_path = Path(output_base_dir)
        if not output_path.exists():
            return []

        runs = []
        for item in output_path.iterdir():
            if item.is_dir() and len(item.name) == 5 and item.name.isdigit():
                life_file = item / "life.json"
                if life_file.exists():
                    runs.append(item.name)

        return sorted(runs)

    @staticmethod
    def load_existing_run(
        run_id: str, output_base_dir: str = "output"
    ) -> "CheckpointManager":
        """Load an existing run by ID."""
        checkpoint_manager = CheckpointManager(
            run_id=run_id, output_base_dir=output_base_dir, auto_continue=True
        )
        if checkpoint_manager.life_file.exists():
            checkpoint_manager.load_life_data()
            logger.info(f"Loaded existing run {run_id}")
            return checkpoint_manager
        else:
            raise FileNotFoundError(f"No existing run found with ID {run_id}")

    @staticmethod
    def auto_detect_interrupted_run(
        output_base_dir: str = "output",
    ) -> Optional["CheckpointManager"]:
        """Automatically detect the most recent interrupted run."""
        runs = CheckpointManager.find_existing_runs(output_base_dir)

        for run_id in reversed(runs):  # Check most recent first
            try:
                cm = CheckpointManager.load_existing_run(run_id, output_base_dir)
                status = cm.life_data.get("status", "")
                if status in ["running", "failed"]:
                    logger.info(f"Auto-detected interrupted run: {run_id}")
                    return cm
            except Exception:
                continue

        return None

    def can_continue(self) -> bool:
        """Check if this run can be continued."""
        if not self.life_file.exists():
            return False

        status = self.life_data.get("status", "")
        completed_steps = self.life_data.get("completed_steps", [])
        failed_steps = self.life_data.get("failed_steps", [])

        return status in ["running", "failed"] and (
            (isinstance(completed_steps, list) and len(completed_steps) > 0)
            or (isinstance(failed_steps, list) and len(failed_steps) > 0)
        )

    def should_auto_continue(self) -> bool:
        """Check if this run should automatically continue."""
        return self.auto_continue and self.can_continue()


def list_runs(output_base_dir: str = "output") -> None:
    """List all available runs with their status."""
    runs = CheckpointManager.find_existing_runs(output_base_dir)

    if not runs:
        print("No existing runs found.")
        return

    print(f"\nAvailable Runs in {output_base_dir}/:")
    print(f"{'='*80}")

    for run_id in runs:
        try:
            cm = CheckpointManager.load_existing_run(run_id, output_base_dir)
            summary = cm.get_run_summary()

            print(
                (
                    f"ID: {run_id} | Status: {summary['status'].upper()} | "
                    f"Progress: {summary['progress']} "
                    f"({summary['progress_percent']:.1f}%) | "
                    f"Stage: {summary['current_stage']}"
                )
            )
        except Exception as e:
            print(f"ID: {run_id} | Error loading run: {e}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Simple CLI for testing
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_runs()
    else:
        # Test checkpoint manager
        cm = CheckpointManager()
        cm.setup_run_directory()
        cm.print_status()
