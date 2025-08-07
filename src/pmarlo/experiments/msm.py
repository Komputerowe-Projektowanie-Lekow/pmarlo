import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from ..markov_state_model.markov_state_model import run_complete_msm_analysis

logger = logging.getLogger(__name__)


@dataclass
class MSMConfig:
    trajectory_files: List[str]
    topology_file: str
    output_dir: str = "experiments_output/msm"
    n_clusters: int = 60
    lag_time: int = 20
    feature_type: str = "phi_psi"
    temperatures: List[float] | None = None


def _timestamp_dir(base_dir: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_msm_experiment(config: MSMConfig) -> Dict:
    """
    Runs Stage 3: MSM construction on provided trajectories.
    Returns a dict with key result file paths.
    """
    run_dir = _timestamp_dir(config.output_dir)

    msm = run_complete_msm_analysis(
        trajectory_files=config.trajectory_files,
        topology_file=config.topology_file,
        output_dir=str(run_dir / "msm"),
        n_clusters=config.n_clusters,
        lag_time=config.lag_time,
        feature_type=config.feature_type,
        temperatures=config.temperatures,
    )

    # Persist config and small summary
    summary = {
        "n_states": int(msm.n_states),
        "analysis_dir": str(run_dir / "msm"),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"MSM experiment complete: {run_dir}")
    return {"run_dir": str(run_dir), "summary": summary}
