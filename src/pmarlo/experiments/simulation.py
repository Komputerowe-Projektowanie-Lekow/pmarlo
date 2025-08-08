import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..pipeline import Pipeline
from .benchmark_utils import (
    build_baseline_object,
    compute_threshold_comparison,
    get_environment_info,
    initialize_baseline_if_missing,
    update_trend,
)
from .kpi import (
    RuntimeMemoryTracker,
    build_benchmark_record,
    compute_conformational_coverage,
    compute_detailed_balance_mad,
    compute_frames_per_second,
    compute_row_stochasticity_mad,
    compute_spectral_gap,
    compute_stationary_entropy,
    compute_transition_matrix_accuracy,
    compute_wall_clock_per_step,
    default_kpi_metrics,
    write_benchmark_json,
)
from .utils import timestamp_dir

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    pdb_file: str
    output_dir: str = "experiments_output/simulation"
    steps: int = 500
    temperature: float = 300.0
    n_states: int = 40
    use_metadynamics: bool = True


def run_simulation_experiment(config: SimulationConfig) -> Dict:
    """
    Runs Stage 1: protein preparation and single-temperature simulation+equilibration
    using the existing Pipeline with use_replica_exchange=False.
    Returns a dict with artifact paths and quick metrics.
    """
    run_dir = timestamp_dir(config.output_dir)

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

    # Prepare and run production with KPI tracking
    errors: list[str] = []
    with RuntimeMemoryTracker() as tracker:
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

    # Standardized input description
    input_desc = {
        "parameters": asdict(config),
        "description": "Single-T simulation input",
    }
    with open(run_dir / "input.json", "w") as f:
        json.dump(input_desc, f, indent=2)

    # KPI benchmark JSON
    conformational_coverage = compute_conformational_coverage(
        states.tolist() if isinstance(states, np.ndarray) else states, config.n_states
    )
    # Quick MSM construction for transition matrix accuracy from states
    transition_matrix_accuracy = None
    row_stochasticity_mad = None
    spectral_gap = None
    stationary_entropy = None
    detailed_balance_mad = None
    ck_mse_factor2 = None
    try:
        # Build a simple row-stochastic transition matrix from states with adaptive lag.
        # Use smaller lag for very short sequences to ensure diagnostics are available in tests.
        if isinstance(states, np.ndarray) and states.size >= 2:
            # Choose a lag that is at least 1 and leaves transitions to count
            tau = max(1, min(20, int(states.size // 3)))
            n_states = int(np.max(states) + 1)
            counts = np.zeros((n_states, n_states), dtype=float)
            for i in range(0, max(0, len(states) - tau)):
                si = int(states[i])
                sj = int(states[i + tau])
                counts[si, sj] += 1.0
            row_sums = counts.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            T = counts / row_sums[:, None]
            transition_matrix_accuracy = compute_transition_matrix_accuracy(T)
            row_stochasticity_mad = compute_row_stochasticity_mad(T)
            spectral_gap = compute_spectral_gap(T)

            # Stationary distribution for entropy and detailed balance checks
            try:
                evals, evecs = np.linalg.eig(T.T)
                idx = int(np.argmax(np.real(evals)))
                pi = np.real(evecs[:, idx])
                pi = np.abs(pi) / max(np.sum(np.abs(pi)), 1e-12)
                stationary_entropy = compute_stationary_entropy(pi)
                detailed_balance_mad = compute_detailed_balance_mad(T, pi)
            except Exception:
                stationary_entropy = None
                detailed_balance_mad = None

            # CK test at factor 2: compare T^2 vs empirical at 2*tau
            try:
                T2_theory = T @ T
                lag2 = 2 * tau
                counts2 = np.zeros((n_states, n_states), dtype=float)
                if len(states) > lag2:
                    for i in range(0, len(states) - lag2):
                        si = int(states[i])
                        sj = int(states[i + lag2])
                        counts2[si, sj] += 1.0
                    row2 = counts2.sum(axis=1)
                    row2[row2 == 0] = 1.0
                    T2_emp = counts2 / row2[:, None]
                    diff = T2_theory - T2_emp
                    ck_mse_factor2 = float(np.mean(diff * diff))
            except Exception:
                ck_mse_factor2 = None
    except Exception:
        transition_matrix_accuracy = None

    kpis = default_kpi_metrics(
        conformational_coverage=conformational_coverage,
        transition_matrix_accuracy=transition_matrix_accuracy,
        replica_exchange_success_rate=None,  # Not applicable for single simulation
        runtime_seconds=tracker.runtime_seconds,
        memory_mb=tracker.max_rss_mb,
    )
    # Enrich input parameters with environment and reproducibility
    # Derive optional integer for number of frames with safe type narrowing
    _num_frames_obj = metrics.get("num_frames")
    num_frames_opt: Optional[int]
    if isinstance(_num_frames_obj, int):
        num_frames_opt = _num_frames_obj
    else:
        num_frames_opt = None
    enriched_input = {
        **asdict(config),
        # Derived throughput KPIs to aid comparison
        "frames_per_second": compute_frames_per_second(
            num_frames_opt, tracker.runtime_seconds
        ),
        "seconds_per_step": compute_wall_clock_per_step(
            tracker.runtime_seconds, config.steps
        ),
        # MSM quick-diagnostics
        "row_stochasticity_mad": row_stochasticity_mad,
        "spectral_gap": spectral_gap,
        "stationary_entropy": stationary_entropy,
        "detailed_balance_mad": detailed_balance_mad,
        "ck_mse_factor2": ck_mse_factor2,
        # Environment and reproducibility
        **get_environment_info(),
        "seed": None,
        "num_frames": metrics.get("num_frames"),
        "num_exchange_attempts": None,
    }

    record = build_benchmark_record(
        algorithm="simulation",
        experiment_id=run_dir.name,
        input_parameters=enriched_input,
        kpi_metrics=kpis,
        notes="Single-T simulation run",
        errors=errors,
    )
    write_benchmark_json(run_dir, record)

    # Baseline and trend; use experiment root (config.output_dir)
    root_dir = Path(config.output_dir)
    baseline_object = build_baseline_object(
        input_parameters=enriched_input,
        results=kpis,
    )
    initialize_baseline_if_missing(root_dir, baseline_object)
    update_trend(root_dir, baseline_object)

    # Write comparison.json against previous trend item if present
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

    logger.info(f"Simulation experiment complete: {run_dir}")
    return {"run_dir": str(run_dir), "metrics": metrics}
