import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from ..markov_state_model.enhanced_msm import run_complete_msm_analysis
from .benchmark_utils import (
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
from .utils import set_seed, timestamp_dir

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
    stride: int = 1
    atom_selection: str | None = None
    seed: int | None = None


def _create_run_directory(output_dir: str) -> Path:
    """Create a new timestamped run directory under the given output path."""
    return timestamp_dir(output_dir)


def _perform_msm_analysis_with_tracking(config: MSMConfig, run_dir: Path):
    """Run MSM analysis while tracking runtime and memory usage."""
    with RuntimeMemoryTracker() as tracker:
        msm = run_complete_msm_analysis(
            trajectory_files=config.trajectory_files,
            topology_file=config.topology_file,
            output_dir=str(run_dir / "msm"),
            n_states=config.n_clusters,
            lag_time=config.lag_time,
            feature_type=config.feature_type,
            temperatures=config.temperatures,
            stride=config.stride,
            atom_selection=config.atom_selection,
        )
    return msm, tracker


def _persist_config_and_summary(config: MSMConfig, msm, run_dir: Path) -> Dict:
    """Persist the configuration and a short summary to JSON files."""
    summary = {
        "n_states": int(msm.n_states),
        "analysis_dir": str(run_dir / "msm"),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _write_input_description(config: MSMConfig, run_dir: Path) -> None:
    """Write a standardized input description JSON for the run."""
    input_desc = {
        "parameters": asdict(config),
        "description": "MSM analysis input",
    }
    with open(run_dir / "input.json", "w", encoding="utf-8") as f:
        json.dump(input_desc, f, indent=2)


def _extract_dtrajs_and_frame_count(msm) -> tuple[list[int], int]:
    """Flatten discrete trajectories if available and count total frames."""
    try:
        dtrajs: list[int] = []
        total_frames = 0
        if hasattr(msm, "dtrajs") and msm.dtrajs:
            for arr in msm.dtrajs:
                try:
                    seq = list(arr)
                    dtrajs.extend(seq)
                    total_frames += len(seq)
                except Exception:
                    # Ignore non-iterable or malformed arrays
                    pass
    except Exception:
        dtrajs = []
        total_frames = 0
    return dtrajs, total_frames


def _build_kpis(dtrajs: list[int], msm, tracker) -> Dict:
    """Compute core KPI metrics for the MSM analysis."""
    return default_kpi_metrics(
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


def _compute_msm_diagnostics(msm) -> Dict:
    """Compute diagnostic statistics from the MSM object."""
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
    # Macrostate CK test (factors 2,3) if available
    ck_macro = None
    try:
        ck_macro = msm.compute_ck_test_macrostates(n_macrostates=3, factors=[2, 3])
    except Exception:
        ck_macro = None
    return {
        "spectral_gap": spectral_gap,
        "stationary_entropy": stationary_entropy,
        "row_stochasticity_mad": row_stochasticity_mad,
        "detailed_balance_mad": detailed_balance_mad,
        "its_convergence_score": its_convergence_score,
        "ck_macro": ck_macro,
    }


def _compute_ck_test_mse(msm) -> float | None:
    """
    Compute CK test MSE at factor 2 by comparing T^2 with the empirical
    transition matrix at 2*lag.
    """
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
            T_array = np.asarray(T, dtype=float)
            T2_theory = T_array @ T_array

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

            diff = T2_theory - T2_emp
            return float(np.mean(diff * diff))
    except Exception:
        return None
    return None


def _build_enriched_input(
    config: MSMConfig,
    msm,
    tracker,
    total_frames: int,
    diagnostics: Dict,
    ck_mse_factor2: float | None,
) -> Dict:
    """Build enriched input metadata that accompanies the benchmark record."""
    return {
        **asdict(config),
        **get_environment_info(),
        "n_states": int(msm.n_states),
        "spectral_gap": diagnostics["spectral_gap"],
        "stationary_entropy": diagnostics["stationary_entropy"],
        "row_stochasticity_mad": diagnostics["row_stochasticity_mad"],
        "detailed_balance_mad": diagnostics["detailed_balance_mad"],
        "its_convergence_score": diagnostics["its_convergence_score"],
        "ck_mse_factor2": ck_mse_factor2,
        "num_frames": int(total_frames),
        "frames_per_second": compute_frames_per_second(
            int(total_frames) if isinstance(total_frames, int) else None,
            getattr(tracker, "runtime_seconds", None),
        ),
        "seconds_per_step": None,
        "num_exchange_attempts": None,
        "overall_acceptance_rate": None,
        "seed": config.seed,
    }


def _write_benchmark_json(
    run_dir: Path, experiment_id: str, enriched_input: Dict, kpis: Dict
) -> None:
    """Create and write the benchmark record JSON file."""
    record = build_benchmark_record(
        algorithm="msm",
        experiment_id=experiment_id,
        input_parameters=enriched_input,
        kpi_metrics=kpis,
        notes="MSM analysis run",
        errors=[],
    )
    write_benchmark_json(run_dir, record)


def _update_baseline_and_trend(
    root_dir: Path, enriched_input: Dict, kpis: Dict
) -> None:
    """Update baseline and trend artifacts at the MSM root directory."""
    baseline_object = build_msm_baseline_object(
        input_parameters=enriched_input,
        results=kpis,
    )
    initialize_baseline_if_missing(root_dir, baseline_object)
    update_trend(root_dir, baseline_object)


def _compare_with_previous_trend_entry(root_dir: Path, run_dir: Path) -> None:
    """Compare the latest trend entry with the previous one and persist diff."""
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
        # Comparison is best-effort; ignore failures
        pass


def run_msm_experiment(config: MSMConfig) -> Dict:
    """
    Run Stage 3: MSM construction on provided trajectories and write outputs.

    Returns a dictionary pointing to the run directory and summary file content.
    """
    set_seed(config.seed)
    run_dir = _create_run_directory(config.output_dir)

    msm, tracker = _perform_msm_analysis_with_tracking(config, run_dir)

    summary = _persist_config_and_summary(config, msm, run_dir)
    _write_input_description(config, run_dir)

    dtrajs, total_frames = _extract_dtrajs_and_frame_count(msm)
    kpis = _build_kpis(dtrajs, msm, tracker)

    diagnostics = _compute_msm_diagnostics(msm)
    ck_mse_factor2 = _compute_ck_test_mse(msm)

    enriched_input = _build_enriched_input(
        config=config,
        msm=msm,
        tracker=tracker,
        total_frames=total_frames,
        diagnostics=diagnostics,
        ck_mse_factor2=ck_mse_factor2,
    )

    _write_benchmark_json(
        run_dir=run_dir,
        experiment_id=run_dir.name,
        enriched_input=enriched_input,
        kpis=kpis,
    )

    root_dir = Path(config.output_dir)
    _update_baseline_and_trend(root_dir, enriched_input, kpis)
    _compare_with_previous_trend_entry(root_dir, run_dir)

    logger.info(f"MSM experiment complete: {run_dir}")
    return {"run_dir": str(run_dir), "summary": summary}
