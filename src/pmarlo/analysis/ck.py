"""Chapman–Kolmogorov test utilities.

This module provides a light-weight CK test implementation that first attempts
an analysis on PCCA+ macrostates.  When the macro decomposition is not
available or does not meet minimal sampling criteria, the test falls back to
microstates restricted to the most populated states.

The main entry point is :func:`run_ck` which computes mean-squared error (MSE)
between predicted and empirical transition matrices for multiples of the base
lag time.  Results are saved as both a plot and small CSV/JSON files mapping
lag multiples to MSE values.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ..states.msm_bridge import pcca_like_macrostates as _pcca_like


@dataclass
class CKRunResult:
    """Container for CK run information."""

    mse: Dict[int, float] = field(default_factory=dict)
    mode: str = "micro"
    insufficient_k: List[int] = field(default_factory=list)


def _row_normalize(C: NDArray[np.float64]) -> NDArray[np.float64]:
    rows: NDArray[np.float64] = C.sum(axis=1).astype(np.float64, copy=False)
    rows[rows == 0] = 1.0
    return (C / rows[:, None]).astype(np.float64, copy=False)


def _count_transitions(
    dtrajs: Sequence[np.ndarray], n_states: int, lag: int
) -> NDArray[np.float64]:
    C: NDArray[np.float64] = np.zeros((n_states, n_states), dtype=np.float64)
    for traj in dtrajs:
        if traj.size <= lag:
            continue
        for i in range(traj.size - lag):
            a = int(traj[i])
            b = int(traj[i + lag])
            if a < 0 or b < 0 or a >= n_states or b >= n_states:
                continue
            C[a, b] += 1.0
    return C


def _largest_connected_indices(C: NDArray[np.float64]) -> NDArray[np.int_]:
    """Return indices of states with at least one observed transition."""
    pops: NDArray[np.float64] = C.sum(axis=1) + C.sum(axis=0)
    return np.where(pops > 0)[0].astype(np.int_)


def _select_top_n_states(C: np.ndarray, n: int) -> np.ndarray:
    pops = C.sum(axis=1) + C.sum(axis=0)
    if np.count_nonzero(pops) == 0:
        return np.array([], dtype=int)
    order = np.argsort(-pops)
    return order[: min(n, len(order))]


def _eigen_gap(T: np.ndarray, k: int) -> float:
    try:
        vals = np.linalg.eigvals(T)
        vals = np.real(vals)
        idx = np.argsort(-vals)
        vals = vals[idx]
        if len(vals) <= k:
            return 0.0
        return float(vals[k - 1] - vals[k])
    except Exception:
        return 0.0


def run_ck(
    dtrajs: Sequence[np.ndarray],
    lag_time: int,
    output_dir: str | Path,
    macro_k: int = 4,
    min_trans: int = 50,
    top_n_micro: int = 50,
    factors: Iterable[int] = (2, 3, 4, 5),
) -> CKRunResult:
    """Run Chapman–Kolmogorov analysis with macro → micro fallback.

    Parameters
    ----------
    dtrajs:
        Sequence of discrete trajectories.
    lag_time:
        Base lag time in frames.
    output_dir:
        Directory to store plot and numeric results.
    macro_k:
        Number of macrostates to attempt via PCCA+.
    min_trans:
        Minimum required transitions per state/macro.
    top_n_micro:
        If macro analysis fails, restrict micro CK to this many most populated
        states.
    factors:
        Lag multiples to evaluate.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = CKRunResult()
    factors = [int(f) for f in factors if int(f) > 1]

    # ------------------------------------------------------------------
    # Compute microstate counts and transition matrix at base lag
    # ------------------------------------------------------------------
    if not dtrajs:
        result.insufficient_k = list(factors)
        _save_ck_outputs(result, out)
        _plot_ck(result, out / "ck.png")
        return result

    n_states = int(max(int(np.max(dt)) for dt in dtrajs) + 1)
    C1_micro = _count_transitions(dtrajs, n_states, lag_time)
    idx_active = _largest_connected_indices(C1_micro)
    if idx_active.size == 0:
        result.insufficient_k = list(factors)
        _save_ck_outputs(result, out)
        _plot_ck(result, out / "ck.png")
        return result
    state_map = {int(old): i for i, old in enumerate(idx_active)}
    filtered_micro = [
        np.array([state_map[s] for s in traj if s in state_map], dtype=int)
        for traj in dtrajs
    ]
    n_micro = idx_active.size
    C1_micro = _count_transitions(filtered_micro, n_micro, lag_time)
    T1_micro = _row_normalize(C1_micro)

    # ------------------------------------------------------------------
    # Attempt macrostate CK via PCCA+
    # ------------------------------------------------------------------
    macro_labels = None
    if n_micro > macro_k and _eigen_gap(T1_micro, macro_k) > 0.01:
        try:
            macro_labels = _pcca_like(T1_micro, n_macrostates=int(macro_k))
        except Exception:
            macro_labels = None
    if macro_labels is not None:
        n_macro = int(np.max(macro_labels)) + 1
        macro_trajs = [macro_labels[traj] for traj in filtered_micro]
        C1_macro = _count_transitions(macro_trajs, n_macro, lag_time)
        if np.all(C1_macro.sum(axis=1) >= min_trans):
            T1_macro = _row_normalize(C1_macro)
            _ck_on_trajs(
                macro_trajs,
                T1_macro,
                lag_time,
                factors,
                min_trans,
                result,
            )
            result.mode = "macro"
            _save_ck_outputs(result, out)
            _plot_ck(result, out / "ck.png")
            return result

    # ------------------------------------------------------------------
    # Fallback: microstate CK on top populated states
    # ------------------------------------------------------------------
    top_idx = _select_top_n_states(C1_micro, int(top_n_micro))
    if top_idx.size == 0:
        result.insufficient_k = list(factors)
        _save_ck_outputs(result, out)
        _plot_ck(result, out / "ck.png")
        return result
    mapping = {int(old): i for i, old in enumerate(top_idx)}
    micro_trajs = [
        np.array([mapping[s] for s in traj if s in mapping], dtype=int)
        for traj in filtered_micro
    ]
    n_sel = top_idx.size
    C1 = _count_transitions(micro_trajs, n_sel, lag_time)
    if np.any(C1.sum(axis=1) < min_trans):
        result.insufficient_k = list(factors)
        _save_ck_outputs(result, out)
        _plot_ck(result, out / "ck.png")
        return result
    T1 = _row_normalize(C1)
    _ck_on_trajs(micro_trajs, T1, lag_time, factors, min_trans, result)
    result.mode = "micro"
    _save_ck_outputs(result, out)
    _plot_ck(result, out / "ck.png")
    return result


def _ck_on_trajs(
    trajs: Sequence[np.ndarray],
    T1: np.ndarray,
    lag: int,
    factors: Sequence[int],
    min_trans: int,
    result: CKRunResult,
) -> None:
    n_states = T1.shape[0]
    for f in factors:
        Ck = _count_transitions(trajs, n_states, lag * int(f))
        if np.any(Ck.sum(axis=1) < min_trans):
            result.insufficient_k.append(int(f))
            continue
        Tk_emp = _row_normalize(Ck)
        Tk_theory = np.linalg.matrix_power(T1, int(f))
        diff = Tk_theory - Tk_emp
        result.mse[int(f)] = float(np.mean(diff * diff))


def _plot_ck(result: CKRunResult, path: Path) -> None:
    plt.figure()
    if result.mse:
        ks = sorted(result.mse.keys())
        mses = [result.mse[k] for k in ks]
        plt.plot(ks, mses, marker="o", linestyle="-", label="MSE")
        plt.xlabel("k (lag multiple)")
        plt.ylabel("MSE")
        plt.legend()
    if result.insufficient_k:
        msg = "insufficient transitions for CK at k=" + ",".join(
            str(k) for k in result.insufficient_k
        )
        plt.text(0.5, 0.5, msg, ha="center", va="center", transform=plt.gca().transAxes)
    plt.tight_layout()
    try:
        plt.savefig(path)
    finally:
        plt.close()


def _save_ck_outputs(result: CKRunResult, out: Path) -> None:
    # CSV table
    csv_path = out / "ck_mse.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["k", "mse"])
        for k, v in sorted(result.mse.items()):
            writer.writerow([k, v])
    # JSON metadata
    json_path = out / "ck_mse.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "mode": result.mode,
                "mse": {str(k): v for k, v in result.mse.items()},
                "insufficient_k": result.insufficient_k,
            },
            fh,
            indent=2,
        )
