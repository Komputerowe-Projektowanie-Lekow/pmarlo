import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from ..markov_state_model.markov_state_model import run_complete_msm_analysis
from .benchmark_utils import (
    build_baseline_object,
    build_msm_baseline_object,
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
    compute_its_convergence_score,
    compute_row_stochasticity_mad,
    compute_spectral_gap,
    compute_stationary_entropy,
    compute_transition_matrix_accuracy,
    default_kpi_metrics,
    write_benchmark_json,
)
from .utils import timestamp_dir

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


def run_msm_experiment(config: MSMConfig) -> Dict:
    """
    Runs Stage 3: MSM construction on provided trajectories.
    Returns a dict with key result file paths.
    """
    run_dir = timestamp_dir(config.output_dir)

    with RuntimeMemoryTracker() as tracker:
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

    # Write standardized input description
    input_desc = {"parameters": asdict(config), "description": "MSM analysis input"}
    with open(run_dir / "input.json", "w") as f:
        json.dump(input_desc, f, indent=2)

    # KPI benchmark JSON
    # Flatten dtrajs if present to estimate coverage and count frames
    try:
        dtrajs = []
        total_frames = 0
        if hasattr(msm, "dtrajs") and msm.dtrajs:
            for arr in msm.dtrajs:
                try:
                    seq = list(arr)
                    dtrajs.extend(seq)
                    total_frames += len(seq)
                except Exception:
                    pass
    except Exception:
        dtrajs = []
        total_frames = 0

    kpis = default_kpi_metrics(
        conformational_coverage=compute_conformational_coverage(
            dtrajs, getattr(msm, "n_states", None)
        ),
        transition_matrix_accuracy=compute_transition_matrix_accuracy(
            getattr(msm, "transition_matrix", None)
        ),
        replica_exchange_success_rate=None,
        runtime_seconds=getattr(tracker, "runtime_seconds", None),
        memory_mb=getattr(tracker, "max_rss_mb", None),
    )
    # Enrich input with environment and MSM diagnostics
    spectral_gap = compute_spectral_gap(getattr(msm, "transition_matrix", None))
    stationary_entropy = compute_stationary_entropy(
        getattr(msm, "stationary_distribution", None)
    )
    row_stochasticity_mad = compute_row_stochasticity_mad(
        getattr(msm, "transition_matrix", None)
    )
    detailed_balance_mad = compute_detailed_balance_mad(
        getattr(msm, "transition_matrix", None),
        getattr(msm, "stationary_distribution", None),
    )
    its_convergence_score = compute_its_convergence_score(
        getattr(msm, "implied_timescales", None)
    )

    # CK test (factor 2): compare T^2 vs empirical T at 2*lag
    ck_mse_factor2 = None
    try:
        import numpy as np

        T = getattr(msm, "transition_matrix", None)
        dtrajs_local = getattr(msm, "dtrajs", None)
        n_states_local = getattr(msm, "n_states", None)
        lag_local = getattr(msm, "lag_time", None)
        if (
            T is not None
            and dtrajs_local
            and isinstance(n_states_local, int)
            and isinstance(lag_local, int)
            and n_states_local > 0
        ):
            # Theoretical T^2
            T = np.asarray(T, dtype=float)
            T2_theory = T @ T

            # Empirical T at 2*lag
            lag2 = 2 * int(lag_local)
            counts = np.zeros((n_states_local, n_states_local), dtype=float)
            for arr in dtrajs_local:
                try:
                    seq = list(arr)
                except Exception:
                    seq = []
                for i in range(0, max(0, len(seq) - lag2)):
                    si = int(seq[i])
                    sj = int(seq[i + lag2])
                    if 0 <= si < n_states_local and 0 <= sj < n_states_local:
                        counts[si, sj] += 1.0
            row_sums = counts.sum(axis=1)
            row_sums[row_sums == 0] = 1.0
            T2_emp = counts / row_sums[:, None]

            # MSE between T^2 and T_{2tau}
            diff = T2_theory - T2_emp
            ck_mse_factor2 = float(np.mean(diff * diff))
    except Exception:
        ck_mse_factor2 = None

    enriched_input = {
        **asdict(config),
        **get_environment_info(),
        "n_states": int(msm.n_states),
        "spectral_gap": spectral_gap,
        "stationary_entropy": stationary_entropy,
        "row_stochasticity_mad": row_stochasticity_mad,
        "detailed_balance_mad": detailed_balance_mad,
        "its_convergence_score": its_convergence_score,
        "ck_mse_factor2": ck_mse_factor2,
        # MSM-specific: frames analyzed
        "num_frames": int(total_frames),
        # Not applicable fields
        "frames_per_second": compute_frames_per_second(
            int(total_frames) if isinstance(total_frames, int) else None,
            getattr(tracker, "runtime_seconds", None),
        ),
        "seconds_per_step": None,
        "num_exchange_attempts": None,
        "overall_acceptance_rate": None,
        "seed": None,
    }

    record = build_benchmark_record(
        algorithm="msm",
        experiment_id=run_dir.name,
        input_parameters=enriched_input,
        kpi_metrics=kpis,
        notes="MSM analysis run",
        errors=[],
    )
    write_benchmark_json(run_dir, record)

    # Baseline and trend at MSM root
    root_dir = Path(config.output_dir)
    baseline_object = build_msm_baseline_object(
        input_parameters=enriched_input,
        results=kpis,
    )
    initialize_baseline_if_missing(root_dir, baseline_object)
    update_trend(root_dir, baseline_object)

    # Comparison with previous trend entry
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

    logger.info(f"MSM experiment complete: {run_dir}")
    return {"run_dir": str(run_dir), "summary": summary}
