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
from .utils import set_seed, timestamp_dir

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    pdb_file: str
    output_dir: str = "experiments_output/simulation"
    steps: int = 500
    temperature: float = 300.0
    n_states: int = 40
    use_metadynamics: bool = True
    seed: int | None = None


def _create_run_dir(output_root: str) -> Path:
    """Create and return a timestamped run directory under the given root."""
    return timestamp_dir(output_root)


def _configure_pipeline(config: SimulationConfig, run_dir: Path) -> Pipeline:
    """Instantiate a single-temperature simulation pipeline."""
    return Pipeline(
        pdb_file=config.pdb_file,
        temperatures=[config.temperature],
        steps=config.steps,
        n_states=config.n_states,
        use_replica_exchange=False,
        use_metadynamics=config.use_metadynamics,
        output_dir=str(run_dir),
        auto_continue=False,
        enable_checkpoints=False,
        # Ensure we record frames frequently enough for short tests
        # (propagated to Simulation via Pipeline)
    )


def _setup_protein_with_fallback(pipeline: Pipeline, pdb_path: str) -> None:
    """
    Attempt protein preparation; if an ImportError occurs (e.g., missing
    PDBFixer), fall back to using the provided PDB directly.
    """
    try:
        # Result is not used downstream; preparation sets pipeline.prepared_pdb
        pipeline.setup_protein()
    except ImportError:
        # PDBFixer not available – fall back to using provided PDB directly
        logger.warning(
            (
                "PDBFixer not installed; skipping protein preparation and using "
                "input PDB as prepared.\nInstall with: pip install "
                "'pmarlo[fixer]' to enable preparation."
            )
        )
        pipeline.prepared_pdb = Path(pdb_path)


def _run_simulation_and_extract_states(
    pipeline: Pipeline,
) -> tuple[list[int], str, float, float]:
    """
    Prepare the system, run production, and extract discrete states while
    tracking runtime and memory. Returns (states, trajectory_path, seconds, rss).
    """
    simulation = pipeline.setup_simulation()
    with RuntimeMemoryTracker() as tracker:
        openmm_sim, meta = simulation.prepare_system()
        traj = simulation.run_production(openmm_sim, meta)
        states = simulation.extract_features(traj)
    runtime_seconds = (
        tracker.runtime_seconds if tracker.runtime_seconds is not None else 0.0
    )
    memory_mb = tracker.max_rss_mb if tracker.max_rss_mb is not None else 0.0
    return states, traj, runtime_seconds, memory_mb


def _build_metrics(
    states: list[int] | np.ndarray, traj: str, prepared_pdb: Optional[Path]
) -> Dict:
    """Assemble quick iteration metrics from states and artifacts."""
    num_frames = int(len(states))
    num_states = int(np.max(states) + 1) if len(states) > 0 else 0
    return {
        "num_states": num_states,
        "num_frames": num_frames,
        "trajectory_file": traj,
        "prepared_pdb": str(prepared_pdb) if prepared_pdb is not None else "",
    }


def _persist_run_artifacts(
    run_dir: Path, config: SimulationConfig, metrics: Dict
) -> None:
    """Write config.json, metrics.json, and standardized input.json."""
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    input_desc = {
        "parameters": asdict(config),
        "description": "Single-T simulation input",
    }
    with open(run_dir / "input.json", "w", encoding="utf-8") as f:
        json.dump(input_desc, f, indent=2)


def _compute_transition_diagnostics(
    states: list[int] | np.ndarray,
) -> tuple[
    Optional[np.ndarray],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[int],
]:
    """
    Construct a simple row-stochastic transition matrix at an adaptive lag and
    compute diagnostics. Returns (T, acc, mad, gap, entropy, db_mad, tau).
    """
    T = None
    acc = mad = gap = ent = db_mad = None
    tau_used: Optional[int] = None
    try:
        if isinstance(states, np.ndarray) and states.size >= 2:
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
            acc = compute_transition_matrix_accuracy(T)
            mad = compute_row_stochasticity_mad(T)
            gap = compute_spectral_gap(T)
            # Stationary distribution derived from T^T eigenvector
            try:
                evals, evecs = np.linalg.eig(T.T)
                idx = int(np.argmax(np.real(evals)))
                pi = np.real(evecs[:, idx])
                pi = np.abs(pi) / max(np.sum(np.abs(pi)), 1e-12)
                ent = compute_stationary_entropy(pi)
                db_mad = compute_detailed_balance_mad(T, pi)
            except Exception:
                ent = None
                db_mad = None
            tau_used = tau
    except Exception:
        acc = None
    return T, acc, mad, gap, ent, db_mad, tau_used


def _compute_ck_mse_factor2(
    T: Optional[np.ndarray], states: list[int] | np.ndarray, tau: Optional[int]
) -> Optional[float]:
    """Compute CK test MSE at factor 2 between T^2 and empirical lag 2*tau."""
    if T is None or tau is None:
        return None
    try:
        T2_theory = T @ T
        lag2 = 2 * tau
        n_states = T.shape[0]
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
            return float(np.mean(diff * diff))
    except Exception:
        return None
    return None


def _build_kpis(
    conformational_coverage: Optional[float],
    transition_matrix_accuracy: Optional[float],
    runtime_seconds: float,
    memory_mb: float,
) -> Dict:
    """Build standardized KPI metrics object."""
    return default_kpi_metrics(
        conformational_coverage=conformational_coverage,
        transition_matrix_accuracy=transition_matrix_accuracy,
        replica_exchange_success_rate=None,
        runtime_seconds=runtime_seconds,
        memory_mb=memory_mb,
    )


def _enrich_input(
    config: SimulationConfig,
    metrics: Dict,
    runtime_seconds: float,
    row_stochasticity_mad: Optional[float],
    spectral_gap: Optional[float],
    stationary_entropy: Optional[float],
    detailed_balance_mad: Optional[float],
    ck_mse_factor2: Optional[float],
) -> Dict:
    """Merge configuration, throughput, MSM diagnostics, and environment info."""
    _num_frames_obj = metrics.get("num_frames")
    num_frames_opt: Optional[int] = (
        _num_frames_obj if isinstance(_num_frames_obj, int) else None
    )
    enriched = {
        **asdict(config),
        "frames_per_second": compute_frames_per_second(num_frames_opt, runtime_seconds),
        "seconds_per_step": compute_wall_clock_per_step(runtime_seconds, config.steps),
        "row_stochasticity_mad": row_stochasticity_mad,
        "spectral_gap": spectral_gap,
        "stationary_entropy": stationary_entropy,
        "detailed_balance_mad": detailed_balance_mad,
        "ck_mse_factor2": ck_mse_factor2,
        **get_environment_info(),
        "seed": config.seed,
        "num_frames": metrics.get("num_frames"),
        "num_exchange_attempts": None,
    }
    return enriched


def _write_benchmark(
    run_dir: Path, enriched_input: Dict, kpis: Dict, errors: list[str]
) -> None:
    """Compose and write the benchmark.json record."""
    record = build_benchmark_record(
        algorithm="simulation",
        experiment_id=run_dir.name,
        input_parameters=enriched_input,
        kpi_metrics=kpis,
        notes="Single-T simulation run",
        errors=errors,
    )
    write_benchmark_json(run_dir, record)


def _update_baseline_and_trend(
    root_dir: Path, enriched_input: Dict, kpis: Dict
) -> None:
    """Initialize baseline if needed and append to trend."""
    baseline_object = build_baseline_object(
        input_parameters=enriched_input, results=kpis
    )
    initialize_baseline_if_missing(root_dir, baseline_object)
    update_trend(root_dir, baseline_object)


def _write_comparison_if_available(root_dir: Path, run_dir: Path) -> None:
    """Write comparison.json using the last two entries from trend.json if present."""
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
        # Non-fatal for experiments
        pass


def run_simulation_experiment(config: SimulationConfig) -> Dict:
    """
    Runs Stage 1: protein preparation and single-temperature simulation +
    equilibration using the existing Pipeline with use_replica_exchange=False.
    Returns a dict with artifact paths and quick metrics.
    """
    set_seed(config.seed)
    run_dir = _create_run_dir(config.output_dir)
    pipeline = _configure_pipeline(config, run_dir)
    _setup_protein_with_fallback(pipeline, config.pdb_file)
    states, traj, runtime_seconds, memory_mb = _run_simulation_and_extract_states(
        pipeline
    )
    metrics = _build_metrics(states, traj, pipeline.prepared_pdb)
    _persist_run_artifacts(run_dir, config, metrics)

    conformational_coverage = compute_conformational_coverage(
        states.tolist() if isinstance(states, np.ndarray) else states, config.n_states
    )
    T, tma, mad, gap, ent, db_mad, tau = _compute_transition_diagnostics(states)
    ck2 = _compute_ck_mse_factor2(T, states, tau)
    kpis = _build_kpis(conformational_coverage, tma, runtime_seconds, memory_mb)
    enriched_input = _enrich_input(
        config,
        metrics,
        runtime_seconds,
        mad,
        gap,
        ent,
        db_mad,
        ck2,
    )
    errors: list[str] = []
    _write_benchmark(run_dir, enriched_input, kpis, errors)

    root_dir = Path(config.output_dir)
    _update_baseline_and_trend(root_dir, enriched_input, kpis)
    _write_comparison_if_available(root_dir, run_dir)

    logger.info(f"Simulation experiment complete: {run_dir}")
    return {"run_dir": str(run_dir), "metrics": metrics}
