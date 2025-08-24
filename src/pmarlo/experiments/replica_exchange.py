import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..manager.checkpoint_manager import CheckpointManager
from ..replica_exchange.config import RemdConfig
from ..replica_exchange.replica_exchange import ReplicaExchange, setup_bias_variables
from .benchmark_utils import (
    build_remd_baseline_object,
    compute_threshold_comparison,
    get_environment_info,
    initialize_baseline_if_missing,
    update_trend,
)
from .kpi import (
    RuntimeMemoryTracker,
    build_benchmark_record,
    compute_replica_exchange_success_rate,
    compute_wall_clock_per_step,
    default_kpi_metrics,
    write_benchmark_json,
)
from .utils import set_seed, timestamp_dir

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
    tmin: float = 300.0
    tmax: float = 350.0
    nreplicas: int = 6
    seed: int | None = None


def run_replica_exchange_experiment(config: ReplicaExchangeConfig) -> Dict:
    """
    Runs Stage 2: REMD with multi-temperature replicas from a prepared PDB.
    Returns a dict with exchange statistics and artifact paths.
    """
    set_seed(config.seed)
    run_dir = timestamp_dir(config.output_dir)

    # Minimal checkpointing confined to this experiment run dir
    cm = CheckpointManager(output_base_dir=str(run_dir), auto_continue=False)
    cm.setup_run_directory()

    # Build temperatures if not provided:
    # prefer more replicas and better spacing for acceptance rates
    temps: Optional[List[float]]
    if config.temperatures is None:
        try:
            from ..utils.replica_utils import exponential_temperature_ladder

            temps = exponential_temperature_ladder(
                config.tmin, config.tmax, config.nreplicas
            )
        except Exception:
            # Fallback to simple exponential spacing
            import numpy as _np

            temps = list(
                _np.linspace(
                    float(config.tmin),
                    float(config.tmax),
                    int(max(2, config.nreplicas)),
                )
            )
    else:
        temps = config.temperatures

    if hasattr(ReplicaExchange, "from_config"):
        remd = ReplicaExchange.from_config(
            RemdConfig(
                pdb_file=config.pdb_file,
                temperatures=temps,
                output_dir=str(run_dir / "remd"),
                exchange_frequency=config.exchange_frequency,
                dcd_stride=2000,
                auto_setup=False,
                random_seed=config.seed,
            )
        )
    else:
        # Backward-compatibility for tests that patch ReplicaExchange with a dummy
        remd = ReplicaExchange(
            pdb_file=config.pdb_file,
            temperatures=temps,
            output_dir=str(run_dir / "remd"),
            exchange_frequency=config.exchange_frequency,
            auto_setup=False,
            random_seed=config.seed,
        )

    bias_vars = (
        setup_bias_variables(config.pdb_file) if config.use_metadynamics else None
    )
    # Plan stride before reporters are created
    if hasattr(remd, "plan_reporter_stride"):
        remd.plan_reporter_stride(
            total_steps=config.total_steps,
            equilibration_steps=config.equilibration_steps,
            target_frames=5000,
        )
    remd.setup_replicas(bias_variables=bias_vars)

    # Run with KPI tracking
    with RuntimeMemoryTracker() as tracker:
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

    # Write standardized input description
    input_desc = {
        "parameters": asdict(config),
        "description": "Replica exchange experiment input",
    }
    with open(run_dir / "input.json", "w") as f:
        json.dump(input_desc, f, indent=2)

    # KPI benchmark JSON
    kpis = default_kpi_metrics(
        conformational_coverage=None,
        transition_matrix_accuracy=None,
        replica_exchange_success_rate=compute_replica_exchange_success_rate(stats),
        runtime_seconds=tracker.runtime_seconds,
        memory_mb=tracker.max_rss_mb,
    )
    # Enrich input with environment and REMD specifics
    enriched_input = {
        **asdict(config),
        **get_environment_info(),
        "seconds_per_step": (
            compute_wall_clock_per_step(tracker.runtime_seconds, config.total_steps)
        ),
        "num_exchange_attempts": (
            stats.get("total_exchange_attempts") if isinstance(stats, dict) else None
        ),
        "overall_acceptance_rate": (
            stats.get("overall_acceptance_rate") if isinstance(stats, dict) else None
        ),
        # Not applicable in REMD benchmark
        "frames_per_second": None,
        "spectral_gap": None,
        "row_stochasticity_mad": None,
        "seed": config.seed,
        "num_frames": None,
    }

    record = build_benchmark_record(
        algorithm="replica_exchange",
        experiment_id=run_dir.name,
        input_parameters=enriched_input,
        kpi_metrics=kpis,
        notes="REMD run",
        errors=[],
    )
    write_benchmark_json(run_dir, record)

    # Baseline and trend at REMD root
    root_dir = Path(config.output_dir)
    baseline_object = build_remd_baseline_object(
        input_parameters=enriched_input,
        results=kpis,
    )
    initialize_baseline_if_missing(root_dir, baseline_object)
    update_trend(root_dir, baseline_object)

    # Comparison against previous trend entry
    try:
        trend_path = root_dir / "trend.json"
        if trend_path.exists():
            with open(trend_path, "r", encoding="utf-8") as tf:
                trend = json.load(tf)
            if isinstance(trend, list) and len(trend) >= 2:
                prev = trend[-2]
                curr = trend[-1]
                comparison = compute_threshold_comparison(prev, curr)
                with open(run_dir / "comparison.json", "w", encoding="utf-8") as cf:
                    json.dump(comparison, cf, indent=2)
    except Exception:
        pass

    logger.info(f"Replica exchange experiment complete: {run_dir}")
    return {
        "run_dir": str(run_dir),
        "stats": stats,
        "trajectories_dir": str(run_dir / "remd"),
    }
