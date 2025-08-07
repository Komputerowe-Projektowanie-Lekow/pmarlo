import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..pipeline import Pipeline

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    pdb_file: str
    output_dir: str = "experiments_output/simulation"
    steps: int = 500
    temperature: float = 300.0
    n_states: int = 40
    use_metadynamics: bool = True


def _timestamp_dir(base_dir: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_simulation_experiment(config: SimulationConfig) -> Dict:
    """
    Runs Stage 1: protein preparation and single-temperature simulation+equilibration
    using the existing Pipeline with use_replica_exchange=False.
    Returns a dict with artifact paths and quick metrics.
    """
    run_dir = _timestamp_dir(config.output_dir)

    # Configure a pipeline for single simulation
    pipeline = Pipeline(
        pdb_file=config.pdb_file,
        temperatures=[config.temperature],
        steps=config.steps,
        n_states=config.n_states,
        use_replica_exchange=False,
        use_metadynamics=config.use_metadynamics,
        output_dir=str(run_dir),
        auto_continue=False,
        enable_checkpoints=False,
    )

    # Set up components without running the full pipeline
    try:
        protein = pipeline.setup_protein()
    except ImportError:
        # PDBFixer not available – fall back to using provided PDB directly
        logger.warning(
            "PDBFixer not installed; skipping protein preparation and using input PDB as prepared.\n"
            "Install with: pip install 'pmarlo[fixer]' to enable preparation."
        )
        pipeline.prepared_pdb = Path(config.pdb_file)
        protein = None
    simulation = pipeline.setup_simulation()

    # Prepare and run production
    openmm_sim, meta = simulation.prepare_system()
    traj = simulation.run_production(openmm_sim, meta)
    states = simulation.extract_features(traj)

    # Quick metrics for iteration
    metrics = {
        "num_states": int(np.max(states) + 1) if len(states) > 0 else 0,
        "num_frames": int(len(states)),
        "trajectory_file": traj,
        "prepared_pdb": str(pipeline.prepared_pdb),
    }

    # Persist config and metrics
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Simulation experiment complete: {run_dir}")
    return {"run_dir": str(run_dir), "metrics": metrics}
