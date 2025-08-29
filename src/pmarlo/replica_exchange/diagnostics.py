from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.special import erfcinv


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

    print("Pair  Acc%  Suggested Δβ")
    for i in range(len(temperatures) - 1):
        pair = (i, i + 1)
        att = pair_attempt_counts.get(pair, 0)
        acc = pair_accept_counts.get(pair, 0)
        rate = acc / max(1, att)
        total_attempts += att
        total_accepts += acc

        delta_beta = betas[i + 1] - betas[i]
        # Clamp rate to avoid division by zero when acceptance is perfect
        rate_clamped = float(np.clip(rate, 1e-6, 1 - 1e-6))
        ratio = erfcinv(target_acceptance) / erfcinv(rate_clamped)
        suggested_dbeta = float(delta_beta * ratio)
        pair_stats.append(
            {
                "pair": pair,
                "acceptance": rate,
                "suggested_delta_beta": suggested_dbeta,
            }
        )
        print(f"{pair}  {rate*100:5.1f}%  {suggested_dbeta:10.6f}")

    global_acceptance = total_accepts / max(1, total_attempts)

    avg_dbeta = float(np.mean([p["suggested_delta_beta"] for p in pair_stats]))
    beta_min = float(betas[0])
    beta_max = float(betas[-1])
    n_new = int(round((beta_max - beta_min) / avg_dbeta)) + 1
    n_new = max(2, n_new)
    new_betas = np.linspace(beta_min, beta_max, n_new)
    suggested_temps = (1.0 / new_betas).tolist()

    # Ensure parent directory exists, even if caller passed a custom path
    try:
        out_path = Path(output_json)
        if out_path.parent and not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
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
