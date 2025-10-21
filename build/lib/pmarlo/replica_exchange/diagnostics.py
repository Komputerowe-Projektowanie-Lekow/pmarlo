from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
from scipy import optimize
from scipy.special import erfc, erfcinv, softmax

from pmarlo import constants as const
from pmarlo.utils.path_utils import ensure_directory


@dataclass(frozen=True)
class _PairStats:
    """Per-neighbour acceptance statistics used for ladder retuning."""

    index: int
    pair: tuple[int, int]
    acceptance: float
    clamped_acceptance: float
    delta_beta: float
    sensitivity: float
    initial_delta: float


def _validate_temperature_series(temperatures: Sequence[float]) -> np.ndarray:
    temps = np.asarray(temperatures, dtype=float)
    if temps.ndim != 1:
        temps = temps.ravel()
    if temps.size < 2:
        raise ValueError("At least two temperatures are required")
    temp_diffs = np.diff(temps)
    if temp_diffs.size and np.any(temp_diffs == 0.0):
        raise ValueError("Input temperatures must be strictly monotonic")
    if temp_diffs.size and not (np.all(temp_diffs > 0.0) or np.all(temp_diffs < 0.0)):
        raise ValueError("Input temperatures must be strictly monotonic")
    return temps


def _collect_pair_records(
    delta_beta_magnitudes: np.ndarray,
    pair_attempt_counts: Mapping[tuple[int, int], int],
    pair_accept_counts: Mapping[tuple[int, int], int],
    *,
    erfc_target: float,
) -> tuple[list[_PairStats], int, int]:
    records: list[_PairStats] = []
    total_attempts = 0
    total_accepts = 0

    for idx, delta_beta in enumerate(delta_beta_magnitudes):
        pair = (idx, idx + 1)
        attempts = int(pair_attempt_counts.get(pair, 0))
        accepts = int(pair_accept_counts.get(pair, 0))
        rate = accepts / max(1, attempts)
        rate_clamped = float(
            np.clip(rate, const.NUMERIC_MIN_RATE, const.NUMERIC_MAX_RATE)
        )
        erfc_observed = erfcinv(rate_clamped)
        sensitivity = erfc_observed / max(delta_beta, const.NUMERIC_MIN_POSITIVE)
        initial_delta = float(
            delta_beta * erfc_target / max(erfc_observed, const.NUMERIC_MIN_POSITIVE)
        )
        records.append(
            _PairStats(
                index=idx,
                pair=pair,
                acceptance=rate,
                clamped_acceptance=rate_clamped,
                delta_beta=float(delta_beta),
                sensitivity=float(sensitivity),
                initial_delta=initial_delta,
            )
        )
        total_attempts += attempts
        total_accepts += accepts

    return records, total_attempts, total_accepts


def _compute_initial_weights(
    initial_deltas: np.ndarray, total_span: float
) -> np.ndarray:
    initial_sum = float(initial_deltas.sum())
    if initial_sum <= 0.0:
        scaled_initial = np.full_like(
            initial_deltas, total_span / max(1, len(initial_deltas))
        )
    else:
        scaled_initial = initial_deltas * (total_span / initial_sum)

    weights = scaled_initial / max(
        const.NUMERIC_MIN_POSITIVE, float(np.sum(scaled_initial))
    )
    weights = np.clip(weights, const.NUMERIC_MIN_POSITIVE, None)
    weights /= float(np.sum(weights))
    return weights


def _solve_spacing(
    sensitivities: np.ndarray,
    target_vec: np.ndarray,
    total_span: float,
    weights0: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, optimize.OptimizeResult, Callable[[np.ndarray], np.ndarray]
]:
    def _params_to_deltas(params: np.ndarray) -> np.ndarray:
        weights = softmax(params)
        return total_span * weights

    def residuals(params: np.ndarray) -> np.ndarray:
        deltas = _params_to_deltas(params)
        predicted = erfc(sensitivities * deltas)
        return predicted - target_vec

    params0 = np.log(weights0)
    lsq = optimize.least_squares(residuals, params0, method="trf")
    optimized_params = params0
    if lsq.success and np.all(np.isfinite(lsq.x)):
        optimized_params = np.asarray(lsq.x, dtype=float)

    optimized_deltas = _params_to_deltas(optimized_params)
    predicted_acceptance = erfc(sensitivities * optimized_deltas)
    return optimized_deltas, predicted_acceptance, lsq, residuals


def _summarise_pair_statistics(
    records: Sequence[_PairStats],
    initial_deltas: np.ndarray,
    optimized_deltas: np.ndarray,
    predicted_acceptance: np.ndarray,
    target_acceptance: float,
) -> list[Dict[str, Any]]:
    pair_stats: list[Dict[str, Any]] = []
    for record, init_delta, opt_delta, pred_rate in zip(
        records, initial_deltas, optimized_deltas, predicted_acceptance
    ):
        pair_stats.append(
            {
                "pair": record.pair,
                "acceptance": record.acceptance,
                "initial_delta_beta_estimate": float(init_delta),
                "suggested_delta_beta": float(opt_delta),
                "predicted_acceptance": float(pred_rate),
                "residual": float(pred_rate - target_acceptance),
            }
        )
        print(
            f"{record.pair}  {record.acceptance*100:5.1f}%"
            f"  {init_delta:11.6f}  {opt_delta:12.6f}  {pred_rate*100:7.3f}%"
        )
    return pair_stats


def compute_exchange_statistics(
    exchange_history: List[List[int]],
    n_replicas: int,
    pair_attempt_counts: Dict[tuple[int, int], int],
    pair_accept_counts: Dict[tuple[int, int], int],
) -> Dict[str, Any]:
    if not exchange_history:
        return {}

    replica_visits = np.zeros((n_replicas, n_replicas))
    for states in exchange_history:
        for replica, state in enumerate(states):
            replica_visits[replica, state] += 1

    replica_probs = replica_visits / len(exchange_history)

    round_trip_times: List[int] = []
    for replica in range(n_replicas):
        start_state = exchange_history[0][replica]
        current_state = start_state
        trip_start = 0
        for step, states in enumerate(exchange_history):
            if states[replica] != current_state:
                current_state = states[replica]
                if current_state == start_state and step > trip_start:
                    round_trip_times.append(step - trip_start)
                    trip_start = step

    per_pair_acceptance = {}
    for k, att in pair_attempt_counts.items():
        acc = pair_accept_counts.get(k, 0)
        rate = acc / max(1, att)
        per_pair_acceptance[f"{k}"] = rate

    return {
        "replica_state_probabilities": replica_probs.tolist(),
        "average_round_trip_time": (
            float(np.mean(round_trip_times)) if round_trip_times else 0.0
        ),
        "round_trip_times": round_trip_times[:10],
        "per_pair_acceptance": per_pair_acceptance,
    }


def compute_diffusion_metrics(
    exchange_history: List[List[int]],
    exchange_frequency_steps: int,
    *,
    spark_max_points: int = 200,
) -> Dict[str, Any]:
    """Compute replica index diffusion metrics from exchange history.

    Metrics:
    - mean_abs_disp_per_sweep: average |Δstate| across replicas per exchange sweep
    - mean_abs_disp_per_10k_steps: scaled by 10k / exchange_frequency_steps
    - sparkline: sampled per-sweep average displacements for plotting
    """
    if not exchange_history or len(exchange_history) < 2:
        return {
            "mean_abs_disp_per_sweep": 0.0,
            "mean_abs_disp_per_10k_steps": 0.0,
            "sparkline": [],
        }
    # per-sweep mean absolute displacement across replicas
    per_sweep: List[float] = []
    for prev, cur in zip(exchange_history[:-1], exchange_history[1:]):
        a = np.asarray(prev, dtype=int)
        b = np.asarray(cur, dtype=int)
        m = float(np.mean(np.abs(b - a)))
        per_sweep.append(m)
    mean_per_sweep = float(np.mean(per_sweep)) if per_sweep else 0.0
    scale = 10000.0 / max(1, int(exchange_frequency_steps))
    per_10k = mean_per_sweep * scale
    # Downsample sparkline to at most spark_max_points
    spark = per_sweep
    if len(spark) > spark_max_points:
        idx = np.linspace(0, len(spark) - 1, spark_max_points).astype(int)
        spark = [float(spark[i]) for i in idx]
    return {
        "mean_abs_disp_per_sweep": mean_per_sweep,
        "mean_abs_disp_per_10k_steps": float(per_10k),
        "sparkline": [float(x) for x in spark],
    }


def retune_temperature_ladder(
    temperatures: List[float],
    pair_attempt_counts: Dict[tuple[int, int], int],
    pair_accept_counts: Dict[tuple[int, int], int],
    target_acceptance: float = 0.30,
    output_json: str = "output/temperatures_suggested.json",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Suggest a new temperature ladder based on pairwise acceptance."""

    temps = _validate_temperature_series(temperatures)
    betas = 1.0 / temps
    delta_betas = np.diff(betas)
    if np.any(delta_betas == 0.0):
        raise ValueError("Input temperatures must be strictly monotonic")
    delta_beta_magnitudes = np.abs(delta_betas)

    print("Pair  Acc%  Initial dBeta  Optimized dBeta  Pred Acc%")
    target_acceptance_clamped = float(
        np.clip(target_acceptance, const.NUMERIC_MIN_RATE, const.NUMERIC_MAX_RATE)
    )
    erfc_target = erfcinv(target_acceptance_clamped)

    records, total_attempts, total_accepts = _collect_pair_records(
        delta_beta_magnitudes,
        pair_attempt_counts,
        pair_accept_counts,
        erfc_target=erfc_target,
    )
    if not records:
        raise ValueError("Unable to derive ladder suggestion from inputs")

    sensitivities_arr = np.asarray([rec.sensitivity for rec in records], dtype=float)
    initial_deltas_arr = np.asarray([rec.initial_delta for rec in records], dtype=float)

    global_acceptance = total_accepts / max(1, total_attempts)
    beta_start = float(betas[0])
    beta_end = float(betas[-1])
    total_span = float(np.sum(delta_beta_magnitudes))

    initial_weights = _compute_initial_weights(initial_deltas_arr, total_span)
    target_vec = np.full_like(sensitivities_arr, target_acceptance_clamped)
    optimized_deltas, predicted_acceptance, lsq, residuals = _solve_spacing(
        sensitivities_arr, target_vec, total_span, initial_weights
    )

    median_delta = float(np.median(initial_deltas_arr))
    if not np.isfinite(median_delta) or median_delta <= 0.0:
        median_delta = float(total_span / max(1, len(initial_deltas_arr)))
    n_intervals = max(1, int(round(total_span / median_delta)))
    n_points = max(2, n_intervals + 1)
    new_betas = np.linspace(beta_start, beta_end, n_points, dtype=float)
    suggested_temps = (1.0 / new_betas).tolist()

    pair_stats = _summarise_pair_statistics(
        records,
        initial_deltas_arr,
        optimized_deltas,
        predicted_acceptance,
        target_acceptance_clamped,
    )

    try:
        out_path = Path(output_json)
        if out_path.parent:
            ensure_directory(out_path.parent)
    except Exception:
        pass

    with open(str(output_json), "w", encoding="utf-8") as fh:
        json.dump(suggested_temps, fh)

    if dry_run:
        speedup = len(temperatures) / len(suggested_temps)
        print(f"Dry-run: predicted speedup ~ {speedup:.2f}x")

    return {
        "global_acceptance": global_acceptance,
        "suggested_temperatures": suggested_temps,
        "pair_statistics": pair_stats,
        "fit_residual_norm": (
            float(np.linalg.norm(lsq.fun))
            if lsq.success and lsq.fun is not None
            else float(np.linalg.norm(residuals(np.log(initial_weights))))
        ),
    }
