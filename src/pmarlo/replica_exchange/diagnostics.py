from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import optimize
from scipy.special import erfc, erfcinv

from pmarlo import constants as const
from pmarlo.utils.path_utils import ensure_directory


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
    """Suggest a new temperature ladder based on pairwise acceptance.

    Uses the Kofke spacing adjustment to estimate the required inverse-
    temperature spacing for a desired acceptance rate. The function prints a
    table summarising current pairwise acceptance and the suggested ``Δβ`` for
    each neighbour pair and writes a new ladder to ``output_json``.

    Parameters
    ----------
    temperatures:
        Current replica temperatures in Kelvin.
    pair_attempt_counts:
        Mapping of ``(i, j)`` neighbour pairs to attempted exchanges.
    pair_accept_counts:
        Mapping of ``(i, j)`` neighbour pairs to accepted exchanges.
    target_acceptance:
        Desired per-pair acceptance probability (default ``0.30``).
    output_json:
        Path to save the suggested temperatures.
    dry_run:
        If ``True`` only report the expected speed-up without modifying the
        replica count.

    Returns
    -------
    Dict[str, Any]
        Dictionary with global acceptance and suggested temperatures.
    """

    if len(temperatures) < 2:
        raise ValueError("At least two temperatures are required")

    betas = 1.0 / np.asarray(temperatures, dtype=float)
    pair_stats: List[Dict[str, Any]] = []
    total_attempts = 0
    total_accepts = 0

    print("Pair  Acc%  Initial Δβ  Optimized Δβ  Pred Acc%")
    target_acceptance_clamped = float(
        np.clip(target_acceptance, const.NUMERIC_MIN_RATE, const.NUMERIC_MAX_RATE)
    )
    erfc_target = erfcinv(target_acceptance_clamped)
    delta_betas = np.diff(betas)
    if np.any(delta_betas <= 0.0):
        raise ValueError("Input temperatures must be strictly monotonic")

    pair_data: List[Dict[str, Any]] = []
    sensitivities: List[float] = []
    initial_deltas: List[float] = []
    for i in range(len(temperatures) - 1):
        pair = (i, i + 1)
        att = pair_attempt_counts.get(pair, 0)
        acc = pair_accept_counts.get(pair, 0)
        rate = acc / max(1, att)
        total_attempts += att
        total_accepts += acc

        delta_beta = delta_betas[i]
        # Clamp rate to avoid division by zero when acceptance is perfect
        rate_clamped = float(
            np.clip(rate, const.NUMERIC_MIN_RATE, const.NUMERIC_MAX_RATE)
        )
        erfc_observed = erfcinv(rate_clamped)
        sensitivity = erfc_observed / max(delta_beta, const.NUMERIC_MIN_POSITIVE)
        sensitivities.append(sensitivity)
        initial_delta = float(
            delta_beta * erfc_target / max(erfc_observed, const.NUMERIC_MIN_POSITIVE)
        )
        initial_deltas.append(initial_delta)
        pair_data.append(
            {
                "pair": pair,
                "acceptance": rate,
                "rate_clamped": rate_clamped,
            }
        )

    global_acceptance = total_accepts / max(1, total_attempts)

    beta_min = float(betas[0])
    beta_max = float(betas[-1])
    total_span = beta_max - beta_min

    sensitivities_arr = np.asarray(sensitivities, dtype=float)
    initial_deltas_arr = np.asarray(initial_deltas, dtype=float)
    if not len(initial_deltas_arr):
        raise ValueError("Unable to derive ladder suggestion from inputs")

    initial_sum = float(initial_deltas_arr.sum())
    if initial_sum <= 0.0:
        scaled_initial = np.full_like(
            initial_deltas_arr, total_span / max(1, len(initial_deltas_arr))
        )
    else:
        scaled_initial = initial_deltas_arr * (total_span / initial_sum)

    def objective(deltas: np.ndarray) -> float:
        predicted = erfc(sensitivities_arr * deltas)
        return float(np.sum((predicted - target_acceptance_clamped) ** 2))

    lower = np.full_like(scaled_initial, const.NUMERIC_MIN_POSITIVE)
    upper = np.full_like(scaled_initial, np.inf)
    bounds = optimize.Bounds(lower, upper)
    constraint = optimize.LinearConstraint(
        np.ones((1, scaled_initial.size)), total_span, total_span
    )

    result = optimize.minimize(
        objective,
        scaled_initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[constraint],
    )

    optimized_deltas = np.asarray(scaled_initial, dtype=float)
    if result.success and np.all(np.isfinite(result.x)):
        optimized_deltas = np.asarray(result.x, dtype=float)

    predicted_acceptance = erfc(sensitivities_arr * optimized_deltas)

    new_betas = np.concatenate(
        (
            np.asarray([beta_min]),
            beta_min + np.cumsum(optimized_deltas, dtype=float),
        )
    )
    new_betas[-1] = beta_max
    suggested_temps = (1.0 / new_betas).tolist()

    pair_stats = []
    for data, init_delta, opt_delta, pred_rate in zip(
        pair_data, initial_deltas_arr, optimized_deltas, predicted_acceptance
    ):
        pair_stats.append(
            {
                "pair": data["pair"],
                "acceptance": data["acceptance"],
                "initial_delta_beta_estimate": float(init_delta),
                "suggested_delta_beta": float(opt_delta),
                "predicted_acceptance": float(pred_rate),
            }
        )
        print(
            f"{data['pair']}  {data['acceptance']*100:5.1f}%"
            f"  {init_delta:11.6f}  {opt_delta:12.6f}  {pred_rate*100:7.3f}%"
        )

    # Ensure parent directory exists, even if caller passed a custom path
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
        print(f"Dry-run: predicted speedup ≈ {speedup:.2f}x")

    return {
        "global_acceptance": global_acceptance,
        "suggested_temperatures": suggested_temps,
        "pair_statistics": pair_stats,
    }
