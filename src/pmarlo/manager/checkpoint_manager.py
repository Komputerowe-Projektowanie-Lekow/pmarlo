"""
Checkpoint and Resume System for PMARLO
Handles state management for long-running molecular dynamics simulations.
"""

import os
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages checkpoints and resume functionality for all pipeline types."""
    
    def __init__(self, run_id: str = None, output_base_dir: str = "output", 
                 pipeline_steps: List[str] = None, auto_continue: bool = False):
        """
        Initialize checkpoint manager.
        
        Args:
            run_id: Unique identifier for this run (5-digit string)
            output_base_dir: Base directory for all outputs
            pipeline_steps: Custom list of pipeline steps (uses default if None)
            auto_continue: Automatically detect and continue interrupted runs
        """
        self.output_base_dir = Path(output_base_dir)
        self.run_id = run_id or self._generate_run_id()
        self.run_dir = self.output_base_dir / self.run_id
        self.life_file = self.run_dir / "life.json"
        self.state_file = self.run_dir / "state.pkl"
        self.config_file = self.run_dir / "config.json"
        self.auto_continue = auto_continue
        
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
            "trajectory_analysis"
        ]
        
        # Handle auto-continue logic
        if auto_continue and self.life_file.exists():
            logger.info(f"Auto-continuing existing run {self.run_id}")
            self.load_life_data()
        else:
            # Initialize life tracking
            self.life_data = {
                "run_id": self.run_id,
                "created": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "current_stage": "initialization",
                "completed_steps": [],
                "failed_steps": [],
                "total_steps": pipeline_steps or default_steps,
                "status": "running",
                "pipeline_type": "remd" if pipeline_steps is None else "custom"
            }
        
        logger.info(f"Checkpoint Manager initialized for run {self.run_id}")
    
    def _generate_run_id(self) -> str:
        """Generate a unique 5-digit run ID."""
        import random
        import string
        return ''.join(random.choices(string.digits, k=5))
    
    def setup_run_directory(self) -> Path:
        """Create and setup the run directory structure."""
        # Create run directory
        self.run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            "trajectories",
            "analysis"
        ]
        
        for subdir in subdirs:
            (self.run_dir / subdir).mkdir(exist_ok=True)
        
        # Save initial life file
        self.save_life_data()
        
        logger.info(f"Run directory setup complete: {self.run_dir}")
        return self.run_dir
    
    def save_life_data(self):
        """Save current life data to JSON file."""
        self.life_data["last_updated"] = datetime.now().isoformat()
        with open(self.life_file, 'w') as f:
            json.dump(self.life_data, f, indent=2)
    
    def load_life_data(self) -> Dict[str, Any]:
        """Load life data from JSON file."""
        if self.life_file.exists():
            with open(self.life_file, 'r') as f:
                self.life_data = json.load(f)
        return self.life_data
    
    def mark_step_started(self, step_name: str):
        """Mark a step as started."""
        self.life_data["current_stage"] = step_name
        self.life_data["status"] = "running"
        self.save_life_data()
        logger.info(f"Step started: {step_name}")
    
    def mark_step_completed(self, step_name: str, metadata: Dict[str, Any] = None):
        """Mark a step as completed."""
        step_data = {
            "name": step_name,
            "completed_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Remove from failed if it was there
        self.life_data["failed_steps"] = [
            s for s in self.life_data["failed_steps"] 
            if s.get("name") != step_name
        ]
        
        # Add to completed (or update if already there)
        self.life_data["completed_steps"] = [
            s for s in self.life_data["completed_steps"] 
            if s.get("name") != step_name
        ]
        self.life_data["completed_steps"].append(step_data)
        
        self.save_life_data()
        logger.info(f"Step completed: {step_name}")
    
    def mark_step_failed(self, step_name: str, error_msg: str):
        """Mark a step as failed."""
        step_data = {
            "name": step_name,
            "failed_at": datetime.now().isoformat(),
            "error": error_msg
        }
        
        # Remove from completed if it was there
        self.life_data["completed_steps"] = [
            s for s in self.life_data["completed_steps"] 
            if s.get("name") != step_name
        ]
        
        # Add to failed (or update if already there)
        self.life_data["failed_steps"] = [
            s for s in self.life_data["failed_steps"] 
            if s.get("name") != step_name
        ]
        self.life_data["failed_steps"].append(step_data)
        
        self.life_data["status"] = "failed"
        self.save_life_data()
        logger.error(f"Step failed: {step_name} - {error_msg}")
    
    def clear_failed_step(self, step_name: str):
        """Clear a step from failed list (when retrying)."""
        self.life_data["failed_steps"] = [
            s for s in self.life_data["failed_steps"] 
            if s.get("name") != step_name
        ]
        
        # If no more failed steps, update status
        if not self.life_data["failed_steps"]:
            self.life_data["status"] = "running"
        
        self.save_life_data()
        logger.info(f"Cleared failed status for step: {step_name}")
    
    def is_step_completed(self, step_name: str) -> bool:
        """Check if a step has been completed."""
        completed_names = [s.get("name") for s in self.life_data["completed_steps"]]
        return step_name in completed_names
    
    def get_next_step(self) -> Optional[str]:
        """Get the next step to execute."""
        completed_names = [s.get("name") for s in self.life_data["completed_steps"]]
        failed_names = [s.get("name") for s in self.life_data["failed_steps"]]
        
        # First priority: retry failed steps
        if failed_names:
            # Return the most recently failed step to retry
            return failed_names[-1]
        
        # Second priority: continue with next uncompleted step
        # But be smarter about it - find the next logical step after the last completed one
        if completed_names:
            # Find the index of the last completed step
            last_completed_idx = -1
            for i, step in enumerate(self.life_data["total_steps"]):
                if step in completed_names:
                    last_completed_idx = max(last_completed_idx, i)
            
            # Return the next step after the last completed one
            if last_completed_idx + 1 < len(self.life_data["total_steps"]):
                return self.life_data["total_steps"][last_completed_idx + 1]
        else:
            # No steps completed yet, start from the beginning
            return self.life_data["total_steps"][0]
        
        return None  # All steps completed
    
    def save_state(self, state_data: Dict[str, Any]):
        """Save arbitrary state data to pickle file."""
        with open(self.state_file, 'wb') as f:
            pickle.dump(state_data, f)
        logger.info("State data saved to checkpoint")
    
    def load_state(self) -> Dict[str, Any]:
        """Load state data from pickle file."""
        if self.state_file.exists():
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration for this run."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info("Configuration saved")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration for this run."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def copy_input_files(self, files_to_copy: List[str]):
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
        total_steps = len(self.life_data["total_steps"])
        completed_steps = len(self.life_data["completed_steps"])
        failed_steps = len(self.life_data["failed_steps"])
        
        return {
            "run_id": self.run_id,
            "status": self.life_data["status"],
            "current_stage": self.life_data["current_stage"],
            "progress": f"{completed_steps}/{total_steps}",
            "progress_percent": (completed_steps / total_steps) * 100,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "created": self.life_data["created"],
            "last_updated": self.life_data["last_updated"],
            "run_directory": str(self.run_dir)
        }
    
    def print_status(self, verbose: bool = True):
        """Print current run status."""
        if not verbose:
            return self.get_run_summary()
            
        summary = self.get_run_summary()
        
        print(f"\n{'='*60}")
        print(f"CHECKPOINT STATUS - Run ID: {summary['run_id']}")
        print(f"{'='*60}")
        print(f"Status: {summary['status'].upper()}")
        print(f"Current Stage: {summary['current_stage']}")
        print(f"Progress: {summary['progress']} ({summary['progress_percent']:.1f}%)")
        print(f"Directory: {summary['run_directory']}")
        print(f"Last Updated: {summary['last_updated']}")
        
        if self.life_data["completed_steps"]:
            print(f"\nCompleted Steps:")
            for step in self.life_data["completed_steps"]:
                print(f"  ✓ {step['name']} ({step['completed_at']})")
        
        if self.life_data["failed_steps"]:
            print(f"\nFailed Steps:")
            for step in self.life_data["failed_steps"]:
                print(f"  ✗ {step['name']} - {step['error']}")
        
        next_step = self.get_next_step()
        if next_step:
            print(f"\nNext Step: {next_step}")
        else:
            print(f"\nAll steps completed!")
        
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
    def load_existing_run(run_id: str, output_base_dir: str = "output") -> 'CheckpointManager':
        """Load an existing run by ID."""
        checkpoint_manager = CheckpointManager(run_id=run_id, output_base_dir=output_base_dir, auto_continue=True)
        if checkpoint_manager.life_file.exists():
            checkpoint_manager.load_life_data()
            logger.info(f"Loaded existing run {run_id}")
            return checkpoint_manager
        else:
            raise FileNotFoundError(f"No existing run found with ID {run_id}")
    
    @staticmethod
    def auto_detect_interrupted_run(output_base_dir: str = "output") -> Optional['CheckpointManager']:
        """Automatically detect the most recent interrupted run."""
        runs = CheckpointManager.find_existing_runs(output_base_dir)
        
        for run_id in reversed(runs):  # Check most recent first
            try:
                cm = CheckpointManager.load_existing_run(run_id, output_base_dir)
                if cm.life_data["status"] in ["running", "failed"]:
                    logger.info(f"Auto-detected interrupted run: {run_id}")
                    return cm
            except Exception:
                continue
        
        return None
    
    def can_continue(self) -> bool:
        """Check if this run can be continued."""
        return (self.life_file.exists() and 
                self.life_data["status"] in ["running", "failed"] and
                (self.life_data["completed_steps"] or self.life_data["failed_steps"]))
    
    def should_auto_continue(self) -> bool:
        """Check if this run should automatically continue."""
        return self.auto_continue and self.can_continue()

def list_runs(output_base_dir: str = "output"):
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
            
            print(f"ID: {run_id} | Status: {summary['status'].upper()} | "
                  f"Progress: {summary['progress']} ({summary['progress_percent']:.1f}%) | "
                  f"Stage: {summary['current_stage']}")
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