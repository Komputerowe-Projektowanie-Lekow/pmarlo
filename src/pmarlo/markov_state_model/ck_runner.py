from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

# Use non-interactive backend for headless/test environments
try:
    import matplotlib as _mpl  # type: ignore

    backend = str(getattr(_mpl, "get_backend", lambda: "")()).lower()
    if not backend.startswith(("agg", "pdf", "svg")):
        _mpl.use("Agg", force=True)
except Exception:  # pragma: no cover - best effort
    pass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ._msm_utils import pcca_like_macrostates as _pcca_like

logger = logging.getLogger(__name__)


@dataclass
class CKRunResult:
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


def _validate_inputs(
    dtrajs: Sequence[np.ndarray], lag_time: int, factors: Sequence[int]
) -> None:
    if not dtrajs:
        raise ValueError("No trajectories provided for analysis")
    if lag_time <= 0:
        raise ValueError(f"Lag time must be positive, got {lag_time}")
    if not factors:
        raise ValueError("No lag factors provided for analysis")
    invalid_factors = [f for f in factors if f <= 1]
    if invalid_factors:
        raise ValueError(f"All lag factors must be > 1, got {invalid_factors}")
    total_frames = sum(len(traj) for traj in dtrajs)
    if total_frames < 100:
        logger.warning(
            f"Very few total frames ({total_frames}) may lead to unreliable results"
        )
    max_factor = max(factors)
    if lag_time * max_factor >= total_frames:
        logger.warning(
            f"Lag time * max factor ({lag_time * max_factor}) exceeds total frames ({total_frames})"
        )


def _preprocess_trajectories(
    dtrajs: Sequence[np.ndarray],
) -> tuple[int, NDArray[np.int_], Sequence[np.ndarray], NDArray[np.float64]]:
    n_states = int(max(int(np.max(dt)) for dt in dtrajs) + 1)
    C1_micro = _count_transitions(dtrajs, n_states, lag=1)
    idx_active = _largest_connected_indices(C1_micro)
    if idx_active.size == 0:
        logger.warning("No connected states found in trajectories")
        return 0, np.array([], dtype=np.int_), [], np.array([])
    state_map = {int(old): i for i, old in enumerate(idx_active)}
    filtered_trajs = [
        np.array([state_map[s] for s in traj if s in state_map], dtype=int)
        for traj in dtrajs
    ]
    n_micro = idx_active.size
    C1_micro_filtered = _count_transitions(filtered_trajs, n_micro, lag=1)
    T1_micro = _row_normalize(C1_micro_filtered)
    return n_micro, idx_active, filtered_trajs, T1_micro


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
        if int(f) in result.insufficient_k:
            result.insufficient_k.remove(int(f))


def _attempt_macro_analysis(
    filtered_trajs: Sequence[np.ndarray],
    T1_micro: np.ndarray,
    lag_time: int,
    macro_k: int,
    min_trans: int,
    factors: Sequence[int],
    result: CKRunResult,
) -> bool:
    n_micro = T1_micro.shape[0]
    if n_micro <= macro_k or _eigen_gap(T1_micro, macro_k) < 0.01:
        logger.info(
            f"Macro analysis not feasible: {n_micro} microstates, eigenvalue gap too small, or insufficient states"
        )
        return False
    try:
        macro_labels = _pcca_like(T1_micro, n_macrostates=int(macro_k))
    except Exception as e:
        logger.warning(f"PCCA+ decomposition failed: {e}")
        return False
    if macro_labels is None:
        logger.info("PCCA+ decomposition returned None")
        return False
    n_macro = int(np.max(macro_labels)) + 1
    macro_trajs = [macro_labels[traj] for traj in filtered_trajs]
    C1_macro = _count_transitions(macro_trajs, n_macro, lag_time)
    if not np.all(C1_macro.sum(axis=1) >= min_trans):
        logger.info(f"Insufficient transitions in macrostates (min_trans={min_trans})")
        return False
    T1_macro = _row_normalize(C1_macro)
    _ck_on_trajs(macro_trajs, T1_macro, lag_time, factors, min_trans, result)
    result.mode = "macro"
    logger.info(
        f"Successfully performed macrostate CK analysis with {n_macro} macrostates"
    )
    return True


def _perform_micro_analysis(
    filtered_trajs: Sequence[np.ndarray],
    C1_micro: np.ndarray,
    lag_time: int,
    top_n_micro: int,
    min_trans: int,
    factors: Sequence[int],
    result: CKRunResult,
) -> None:
    top_idx = _select_top_n_states(C1_micro, int(top_n_micro))
    if top_idx.size == 0:
        logger.warning("No populated states found for micro analysis")
        return
    mapping = {int(old): i for i, old in enumerate(top_idx)}
    micro_trajs = [
        np.array([mapping[s] for s in traj if s in mapping], dtype=int)
        for traj in filtered_trajs
    ]
    n_sel = top_idx.size
    C1 = _count_transitions(micro_trajs, n_sel, lag_time)
    if np.any(C1.sum(axis=1) < min_trans):
        logger.info(
            f"Insufficient transitions in selected microstates (min_trans={min_trans})"
        )
        return
    T1 = _row_normalize(C1)
    _ck_on_trajs(micro_trajs, T1, lag_time, factors, min_trans, result)
    result.mode = "micro"
    logger.info(f"Successfully performed microstate CK analysis with {n_sel} states")


def _save_ck_outputs(result: CKRunResult, out: Path) -> None:
    csv_path = out / "ck_mse.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["k", "mse"])
        for k, v in sorted(result.mse.items()):
            writer.writerow([k, v])
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


def _save_outputs_and_plot(result: CKRunResult, output_dir: Path) -> None:
    _save_ck_outputs(result, output_dir)
    _plot_ck(result, output_dir / "ck.png")


def run_ck(
    dtrajs: Sequence[np.ndarray],
    lag_time: int,
    output_dir: str | Path,
    macro_k: int = 4,
    min_trans: int = 50,
    top_n_micro: int = 50,
    factors: Iterable[int] = (2, 3, 4, 5),
) -> CKRunResult:
    factors_list = [int(f) for f in factors if int(f) > 1]
    _validate_inputs(dtrajs, lag_time, factors_list)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = CKRunResult()
    result.insufficient_k = factors_list.copy()
    logger.info(
        f"Starting CK analysis with {len(dtrajs)} trajectories, lag_time={lag_time}, factors={factors_list}"
    )
    if not dtrajs:
        _save_outputs_and_plot(result, out)
        return result
    n_micro, idx_active, filtered_trajs, T1_micro = _preprocess_trajectories(dtrajs)
    if n_micro == 0 or idx_active.size == 0:
        logger.warning("No valid states found after preprocessing")
        _save_outputs_and_plot(result, out)
        return result
    C1_micro = _count_transitions(filtered_trajs, n_micro, lag_time)
    macro_success = _attempt_macro_analysis(
        filtered_trajs, T1_micro, lag_time, macro_k, min_trans, factors_list, result
    )
    if macro_success:
        logger.info("Macrostate analysis succeeded")
        _save_outputs_and_plot(result, out)
        return result
    logger.info("Attempting microstate fallback analysis")
    _perform_micro_analysis(
        filtered_trajs, C1_micro, lag_time, top_n_micro, min_trans, factors_list, result
    )
    _save_outputs_and_plot(result, out)
    return result
