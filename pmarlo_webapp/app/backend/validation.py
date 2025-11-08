"""Simulation directory validation and discovery.

This module provides tools to scan, validate, and report on simulation run
directories, detecting issues like missing files, incomplete runs, and
directories not tracked in state.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .run_metadata import load_run_plan

logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """Status categories for simulation runs."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    EMPTY = "empty"
    MISSING_ANALYSIS = "missing_analysis"
    MISSING_DEMUX = "missing_demux"
    MISSING_STATE_ENTRY = "missing_state_entry"
    IN_PROGRESS = "in_progress"
    CORRUPTED = "corrupted"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in a run directory."""

    severity: str
    message: str
    details: Optional[str] = None


@dataclass
class RunValidation:
    """Validation result for a simulation run directory."""

    run_id: str
    run_dir: Path
    status: RunStatus
    in_state: bool
    has_shards: bool
    shard_count: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check if the run is valid for loading and use."""
        return self.status in (
            RunStatus.COMPLETE,
            RunStatus.MISSING_ANALYSIS,
            RunStatus.MISSING_DEMUX,
        )

    @property
    def can_create_shards(self) -> bool:
        """Check if shards can be created from this run."""
        return self.is_valid and not self.has_shards

    def summary(self) -> str:
        """Get a human-readable summary of the validation result."""
        parts = [f"{self.run_id}: {self.status.value}"]
        if not self.in_state:
            parts.append("(not in state)")
        if self.has_shards:
            parts.append(f"({self.shard_count} shards)")
        if self.issues:
            parts.append(f"({len(self.issues)} issues)")
        return " ".join(parts)


class ValidationMixin:
    """Methods for validating and discovering simulation runs.

    This mixin provides comprehensive validation of simulation directories,
    detecting missing files, incomplete runs, and schema issues.
    """

    def discover_all_runs(self) -> List[RunValidation]:
        """Discover and validate all simulation run directories.

        Scans the sims directory for all run folders and validates each one,
        checking for completeness, missing files, and state synchronization.

        Returns
        -------
        List[RunValidation]
            List of validation results for all discovered runs
        """
        if not self.layout.sims_dir.exists():
            logger.warning(f"Sims directory does not exist: {self.layout.sims_dir}")
            return []

        validations = []
        state_run_ids = {str(entry.get("run_id", "")) for entry in self.state.runs}
        shard_run_ids = {str(entry.get("run_id", "")) for entry in self.state.shards}

        # Scan all directories in sims folder
        for run_dir in sorted(self.layout.sims_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            run_id = run_dir.name
            validation = self._validate_run_directory(
                run_id=run_id,
                run_dir=run_dir,
                in_state=run_id in state_run_ids,
                has_shards=run_id in shard_run_ids,
            )
            validations.append(validation)

        logger.info(
            f"Discovered {len(validations)} runs: "
            f"{sum(1 for v in validations if v.is_valid)} valid, "
            f"{sum(1 for v in validations if not v.in_state)} not in state"
        )

        return validations

    def validate_run(self, run_id: str) -> Optional[RunValidation]:
        """Validate a specific run directory.

        Parameters
        ----------
        run_id : str
            The run ID to validate

        Returns
        -------
        Optional[RunValidation]
            Validation result, or None if directory doesn't exist
        """
        run_dir = self.layout.sims_dir / run_id
        if not run_dir.exists():
            return None

        state_run_ids = {str(entry.get("run_id", "")) for entry in self.state.runs}
        shard_run_ids = {str(entry.get("run_id", "")) for entry in self.state.shards}

        return self._validate_run_directory(
            run_id=run_id,
            run_dir=run_dir,
            in_state=run_id in state_run_ids,
            has_shards=run_id in shard_run_ids,
        )

    def _validate_run_directory(
        self,
        run_id: str,
        run_dir: Path,
        in_state: bool,
        has_shards: bool,
    ) -> RunValidation:
        """Validate a single run directory.

        Parameters
        ----------
        run_id : str
            The run identifier
        run_dir : Path
            Path to the run directory
        in_state : bool
            Whether this run is tracked in state
        has_shards : bool
            Whether shards exist for this run

        Returns
        -------
        RunValidation
            Validation result with status and issues
        """
        issues: List[ValidationIssue] = []
        metadata: Dict[str, Any] = {}

        # Check for replica_exchange subdirectory
        remd_dir = run_dir / "replica_exchange"
        if not remd_dir.exists():
            return RunValidation(
                run_id=run_id,
                run_dir=run_dir,
                status=RunStatus.EMPTY,
                in_state=in_state,
                has_shards=has_shards,
                issues=[
                    ValidationIssue(
                        severity="error",
                        message="Missing replica_exchange directory",
                    )
                ],
                metadata=metadata,
            )

        # Check for trajectory files
        traj_files = list(remd_dir.glob("replica_*.dcd"))
        demux_files = list(remd_dir.glob("demux_*.dcd"))

        metadata["trajectory_count"] = len(traj_files)
        metadata["demux_count"] = len(demux_files)

        if not traj_files:
            return RunValidation(
                run_id=run_id,
                run_dir=run_dir,
                status=RunStatus.EMPTY,
                in_state=in_state,
                has_shards=has_shards,
                issues=[
                    ValidationIssue(
                        severity="error",
                        message="No trajectory files found",
                    )
                ],
                metadata=metadata,
            )

        # Validate that trajectory files are readable
        corrupted_files = []
        try:
            import mdtraj as md
            # Find a PDB file to use as topology
            from pathlib import Path
            pdb_candidates = list(Path(self.layout.inputs_dir).glob("*.pdb"))
            if pdb_candidates:
                pdb_path = pdb_candidates[0]
                # Check a sample of trajectory files
                for traj_file in (traj_files[:1] if traj_files else []) + (demux_files[:1] if demux_files else []):
                    try:
                        # Try to read just the first frame to validate the file
                        test_traj = md.load_frame(str(traj_file), 0, top=str(pdb_path))
                        if test_traj is None or test_traj.n_frames == 0:
                            corrupted_files.append(traj_file.name)
                    except (OSError, IOError, ValueError) as e:
                        if "corruption" in str(e).lower() or "could not open" in str(e).lower():
                            corrupted_files.append(traj_file.name)
                            logger.warning(f"Corrupted trajectory file detected: {traj_file.name}: {e}")
        except Exception as e:
            logger.debug(f"Could not validate trajectory files: {e}")

        if corrupted_files:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"Corrupted trajectory files detected: {', '.join(corrupted_files)}",
                    details="These files cannot be read. They may be from an interrupted simulation that needs to be restarted.",
                )
            )
            metadata["corrupted_files"] = corrupted_files

        # Check for checkpoint files (useful for both in-progress and corrupted runs)
        checkpoint_files = list(remd_dir.glob("checkpoint_step_*.pkl"))
        if checkpoint_files:
            metadata["checkpoint_count"] = len(checkpoint_files)
            # Find the latest checkpoint
            checkpoint_numbers = []
            for cp in checkpoint_files:
                try:
                    num = int(cp.stem.split("_")[-1])
                    checkpoint_numbers.append(num)
                except (ValueError, IndexError):
                    pass
            if checkpoint_numbers:
                metadata["latest_checkpoint_step"] = max(checkpoint_numbers)

        # Check if files are corrupted - this takes precedence
        if corrupted_files:
            status = RunStatus.CORRUPTED
            if checkpoint_files:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        message=f"Found {len(checkpoint_files)} checkpoint files - run can be resumed",
                        details=f"Latest checkpoint at step {metadata.get('latest_checkpoint_step', 'unknown')}",
                    )
                )
        else:
            # Check for analysis results
            analysis_file = remd_dir / "analysis_results.json"
            if not analysis_file.exists():
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message="Missing analysis_results.json",
                        details="Run may not have completed analysis step",
                    )
                )

            # Check for demultiplexed trajectories
            if not demux_files:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        message="No demultiplexed trajectories found",
                        details="Run may not have completed demultiplexing",
                    )
                )
                status = RunStatus.MISSING_DEMUX
            elif not analysis_file.exists():
                status = RunStatus.MISSING_ANALYSIS
            else:
                status = RunStatus.COMPLETE

            # Check if this is an in-progress run based on checkpoints
            if checkpoint_files and not analysis_file.exists():
                status = RunStatus.IN_PROGRESS
                issues.append(
                    ValidationIssue(
                        severity="info",
                        message="Run appears to be in progress",
                        details=f"Found {len(checkpoint_files)} checkpoint files at step {metadata.get('latest_checkpoint_step', 'unknown')}",
                    )
                )

        # Check for provenance file
        provenance_file = remd_dir / "provenance.json"
        if provenance_file.exists():
            try:
                with open(provenance_file, "r") as f:
                    provenance = json.load(f)
                    metadata["provenance"] = provenance
                    if "total_steps" in provenance:
                        metadata["total_steps"] = provenance["total_steps"]
                    if "temperatures" in provenance:
                        metadata["temperatures"] = provenance["temperatures"]
            except Exception as e:
                logger.warning(f"Failed to read provenance for {run_id}: {e}")

        # Check for exchange diagnostics
        exchange_diag_file = remd_dir / "exchange_diagnostics.json"
        if exchange_diag_file.exists():
            try:
                with open(exchange_diag_file, "r") as f:
                    diag = json.load(f)
                    if "exchange_accepted" in diag and "exchange_attempts" in diag:
                        attempts = diag["exchange_attempts"]
                        accepted = diag["exchange_accepted"]
                        if attempts > 0:
                            acceptance_rate = accepted / attempts
                            metadata["exchange_acceptance_rate"] = acceptance_rate
            except Exception as e:
                logger.warning(f"Failed to read exchange diagnostics for {run_id}: {e}")

        # Check if run is in state
        if not in_state:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message="Run not tracked in state.json",
                    details="This run will not appear in standard UI dropdowns",
                )
            )

        # Check shard status - always check physical files, not just state
        shard_count = 0
        shard_dir = self.layout.shards_dir / run_id
        physical_shards_exist = False
        if shard_dir.exists():
            shard_files = list(shard_dir.glob("*.json"))
            shard_count = len(shard_files)
            if shard_count > 0:
                physical_shards_exist = True
                metadata["shard_files"] = shard_count
                # Update has_shards based on actual files, not just state tracking
                # This ensures the UI shows correct status even if state.json is out of sync
                if not has_shards:
                    # Shards exist on disk but not tracked in state
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            message=f"Found {shard_count} shard files on disk but not tracked in state.json",
                            details="These shards may be from a previous session. Consider adding them to state or deleting them.",
                        )
                    )
                has_shards = True

        run_plan = load_run_plan(run_dir)
        if run_plan:
            metadata["run_plan"] = run_plan
            config_snapshot = run_plan.get("config")
            if config_snapshot:
                metadata.setdefault("config_snapshot", config_snapshot)

        return RunValidation(
            run_id=run_id,
            run_dir=run_dir,
            status=status,
            in_state=in_state,
            has_shards=has_shards,
            shard_count=shard_count,
            issues=issues,
            metadata=metadata,
        )

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of all runs with validation statistics.

        Returns
        -------
        Dict[str, Any]
            Summary with counts and statistics
        """
        validations = self.discover_all_runs()

        status_counts = {}
        for status in RunStatus:
            status_counts[status.value] = sum(
                1 for v in validations if v.status == status
            )

        return {
            "total_runs": len(validations),
            "in_state": sum(1 for v in validations if v.in_state),
            "not_in_state": sum(1 for v in validations if not v.in_state),
            "has_shards": sum(1 for v in validations if v.has_shards),
            "can_create_shards": sum(1 for v in validations if v.can_create_shards),
            "status_counts": status_counts,
            "total_issues": sum(len(v.issues) for v in validations),
        }

    def get_missing_state_entries(self) -> List[RunValidation]:
        """Get all runs that exist on disk but are not in state.

        Returns
        -------
        List[RunValidation]
            List of validation results for runs not in state
        """
        validations = self.discover_all_runs()
        return [v for v in validations if not v.in_state and v.is_valid]

    def add_run_to_state(self, run_id: str) -> bool:
        """Add a discovered run to the state.

        Parameters
        ----------
        run_id : str
            The run ID to add to state

        Returns
        -------
        bool
            True if successfully added, False otherwise
        """
        validation = self.validate_run(run_id)
        if validation is None or not validation.is_valid:
            logger.warning(f"Cannot add invalid run {run_id} to state")
            return False

        if validation.in_state:
            logger.info(f"Run {run_id} already in state")
            return True

        # Try to reconstruct run metadata from filesystem
        run_dir = self.layout.sims_dir / run_id
        remd_dir = run_dir / "replica_exchange"

        # Find topology file
        pdb_files = list(self.layout.inputs_dir.glob("*.pdb"))
        if not pdb_files:
            logger.error(f"No PDB files found in inputs directory")
            return False
        pdb_path = pdb_files[0]

        # Find trajectory files
        traj_files = sorted(remd_dir.glob("demux_*.dcd"))
        if not traj_files:
            traj_files = sorted(remd_dir.glob("replica_*.dcd"))

        if not traj_files:
            logger.error(f"No trajectory files found for {run_id}")
            return False

        # Extract temperatures from metadata
        temperatures = validation.metadata.get("temperatures", [300.0])
        analysis_temperatures = temperatures

        # Try to parse temperatures from demux filenames
        if not temperatures:
            for traj_file in traj_files:
                if "demux_T" in traj_file.stem:
                    try:
                        temp_str = traj_file.stem.split("_T")[1].replace("K", "")
                        temp = float(temp_str)
                        analysis_temperatures.append(temp)
                    except (IndexError, ValueError):
                        pass

        if not analysis_temperatures:
            analysis_temperatures = [300.0]

        # Create state entry
        metadata = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "pdb": str(pdb_path),
            "temperatures": temperatures or analysis_temperatures,
            "analysis_temperatures": analysis_temperatures,
            "steps": validation.metadata.get("total_steps", 0),
            "quick": False,
            "random_seed": None,
            "traj_files": [str(f) for f in traj_files],
            "created_at": run_dir.stat().st_mtime,
            "stub_result": False,
            "discovered": True,
        }

        self.state.append_run(metadata)
        logger.info(f"Added discovered run {run_id} to state")
        return True

