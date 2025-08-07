import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..manager.checkpoint_manager import CheckpointManager
from ..replica_exchange.replica_exchange import ReplicaExchange, setup_bias_variables

logger = logging.getLogger(__name__)


@dataclass
class ReplicaExchangeConfig:
    pdb_file: str
    output_dir: str = "experiments_output/replica_exchange"
    temperatures: Optional[List[float]] = None  # defaults handled by class
    total_steps: int = 800
    equilibration_steps: int = 200
    exchange_frequency: int = 50
    use_metadynamics: bool = True


def _timestamp_dir(base_dir: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_replica_exchange_experiment(config: ReplicaExchangeConfig) -> Dict:
    """
    Runs Stage 2: REMD with multi-temperature replicas from a prepared PDB.
    Returns a dict with exchange statistics and artifact paths.
    """
    run_dir = _timestamp_dir(config.output_dir)

    # Minimal checkpointing confined to this experiment run dir
    cm = CheckpointManager(output_base_dir=str(run_dir), auto_continue=False)
    cm.setup_run_directory()

    remd = ReplicaExchange(
        pdb_file=config.pdb_file,
        temperatures=config.temperatures,
        output_dir=str(run_dir / "remd"),
        exchange_frequency=config.exchange_frequency,
        auto_setup=False,
    )

    bias_vars = (
        setup_bias_variables(config.pdb_file) if config.use_metadynamics else None
    )
    remd.setup_replicas(bias_variables=bias_vars)

    remd.run_simulation(
        total_steps=config.total_steps,
        equilibration_steps=config.equilibration_steps,
        checkpoint_manager=cm,
    )

    stats = remd.get_exchange_statistics()

    # Persist config and stats
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(run_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Replica exchange experiment complete: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "stats": stats,
        "trajectories_dir": str(run_dir / "remd"),
    }
