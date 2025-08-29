from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import connected_components

from ._base import CKTestResult


class CKMixin:
    # Attributes provided by concrete classes mixing this in
    dtrajs: List[np.ndarray]
    n_states: int
    lag_time: int
    transition_matrix: Optional[np.ndarray]
    output_dir: Any  # Path-like; used for diagnostics persistence
    # Provided by StatesMixin
    _build_macro_trajectories: Callable[
        [List[np.ndarray], np.ndarray], List[np.ndarray]
    ]

    def _bootstrap_counts(
        self, assignments: np.ndarray, n_boot: int = 1000
    ) -> np.ndarray:
        """Bootstrap category counts from state assignments.

        Parameters
        ----------
        assignments:
            1D array of discrete state indices.
        n_boot:
            Number of bootstrap resamples.

        Returns
        -------
        np.ndarray
            Array of shape (n_boot, n_states) with bootstrapped counts.
        """
        import numpy as _np

        if (
            getattr(self, "n_states", None) is None
            or int(getattr(self, "n_states", 0)) <= 0
        ):
            raise ValueError("n_states must be set before bootstrapping counts")
        assignments = _np.asarray(assignments, dtype=int)
        n = int(assignments.size)
        k = int(self.n_states)
        rng = _np.random.default_rng()
        out = _np.zeros((int(n_boot), k), dtype=float)
        for b in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            sample = assignments[idx]
            out[b] = _np.bincount(sample, minlength=k)
        return out

    def compute_ck_test_micro(
        self,
        factors: Optional[List[int]] = None,
        max_states: int = 50,
        min_transitions: int = 5,
    ) -> CKTestResult:
        factors = self._normalize_ck_factors(factors)
        result = CKTestResult(
            mode="micro",
            thresholds={
                "min_transitions_per_state": int(min_transitions),
                "max_states": int(max_states),
            },
        )
        if not self.dtrajs or self.n_states <= 1 or self.lag_time <= 0:
            result.insufficient_data = True
            return result

        T_all, C_all = self._count_micro_T(
            self.dtrajs, self.n_states, int(self.lag_time)
        )
        idx = self._largest_connected_states(C_all, int(max_states))
        if idx.size == 0:
            result.insufficient_data = True
            return result

        state_map = {int(old): i for i, old in enumerate(idx)}
        filtered = [
            np.array([state_map[s] for s in traj if s in state_map], dtype=int)
            for traj in self.dtrajs
        ]
        n_sel = int(len(idx))
        T1, C1 = self._count_micro_T(filtered, n_sel, int(self.lag_time))
        if np.any(C1.sum(axis=1) < min_transitions):
            result.insufficient_data = True
            return result

        for f in factors:
            T_emp, Ck = self._count_micro_T(
                filtered, n_sel, int(self.lag_time) * int(f)
            )
            if np.any(Ck.sum(axis=1) < min_transitions):
                result.insufficient_data = True
                return result
            T_theory = np.linalg.matrix_power(T1, int(f))
            diff = T_theory - T_emp
            result.mse[int(f)] = float(np.mean(diff * diff))
        return result

    def compute_ck_test_macrostates(
        self,
        n_macrostates: int = 3,
        factors: Optional[List[int]] = None,
        min_transitions: int = 5,
    ) -> CKTestResult:
        factors = self._normalize_ck_factors(factors)
        gap = self._micro_eigen_gap(k=2)
        if gap is None or gap <= 0.01:
            # fallback to micro
            return self.compute_ck_test_micro(factors=factors)

        result = CKTestResult(
            mode="macro", thresholds={"min_transitions_per_state": int(min_transitions)}
        )
        if not self.dtrajs or self.n_states <= 0 or self.lag_time <= 0:
            result.insufficient_data = True
            return result

        macro_labels = getattr(self, "_micro_to_macro_labels", None)
        if callable(macro_labels):
            macro_labels = macro_labels(n_macrostates=n_macrostates)
        else:
            macro_labels = None
        if macro_labels is None:
            return self.compute_ck_test_micro(factors=factors)
        n_macros = int(np.max(macro_labels) + 1)
        if n_macros <= 1:
            return self.compute_ck_test_micro(factors=factors)

        macro_trajs = self._build_macro_trajectories(self.dtrajs, macro_labels)
        T1, C1 = self._count_macro_T_and_counts(
            macro_trajs, n_macros, int(self.lag_time)
        )
        if np.any(C1.sum(axis=1) < min_transitions):
            result.insufficient_data = True
            return result

        for f in factors:
            mse, Ck = self._ck_mse_for_factor(
                T1, macro_trajs, n_macros, int(self.lag_time), int(f)
            )
            if mse is None or np.any(Ck.sum(axis=1) < min_transitions):
                result.insufficient_data = True
                return result
            result.mse[int(f)] = float(mse)
        return result

    def select_lag_time_ck(
        self, tau_candidates: List[int], factor: int = 2, mse_epsilon: float = 0.05
    ) -> int:
        taus, mses, its_list = self._evaluate_candidates(
            tau_candidates=tau_candidates, factor=int(factor)
        )
        selected = self._select_tau_from_prefix(
            taus=taus, mses=mses, its_list=its_list, mse_epsilon=float(mse_epsilon)
        )
        selected = self._apply_global_mse_fallback(
            taus=taus, mses=mses, selected=selected
        )
        selected = self._prefer_tau_two_on_tie(taus=taus, mses=mses, selected=selected)

        self.lag_time = int(selected)
        self._persist_ck_diagnostics(taus=taus, mses=mses, selected=selected)
        return int(selected)

    def _evaluate_candidates(
        self, *, tau_candidates: List[int], factor: int
    ) -> Tuple[list[int], list[float], list[float]]:
        taus: list[int] = []
        mses: list[float] = []
        its_list: list[float] = []
        for tau in tau_candidates:
            tau = int(tau)
            taus.append(tau)
            T1, _ = self._count_micro_T(self.dtrajs, self.n_states, tau)
            its_list.append(self._slowest_its_from_T(T1, tau))
            mses.append(self._ck_mse_from_T(T1, tau, int(factor)))
        return taus, mses, its_list

    def _select_tau_from_prefix(
        self,
        *,
        taus: list[int],
        mses: list[float],
        its_list: list[float],
        mse_epsilon: float,
    ) -> int:
        selected = int(taus[0])
        prev_mse: Optional[float] = None
        prev_its: Optional[float] = None
        for t, m, s in zip(taus, mses, its_list):
            if prev_mse is None:
                selected = int(t)
                prev_mse = float(m)
                prev_its = float(s)
                continue
            # Help static type checkers: prev_mse/prev_its are set here
            assert prev_mse is not None and prev_its is not None
            non_decreasing_its = s >= prev_its - 1e-12
            rel_improvement = (prev_mse - m) / max(prev_mse, 1e-12)
            if non_decreasing_its and rel_improvement > float(mse_epsilon):
                selected = int(t)
                prev_mse = float(m)
                prev_its = float(s)
            else:
                break
        return int(selected)

    def _apply_global_mse_fallback(
        self, *, taus: list[int], mses: list[float], selected: int
    ) -> int:
        idx_min = int(np.nanargmin(mses))
        best_tau = int(taus[idx_min])
        if best_tau != selected:
            selected = best_tau
        return int(selected)

    def _prefer_tau_two_on_tie(
        self, *, taus: list[int], mses: list[float], selected: int
    ) -> int:
        if selected == 1 and 2 in taus:
            j = int(taus.index(2))
            idx_min = int(np.nanargmin(mses))
            if mses[j] <= mses[idx_min] + 1e-12:
                selected = 2
        return int(selected)

    def _persist_ck_diagnostics(
        self, *, taus: list[int], mses: list[float], selected: int
    ) -> None:
        try:
            import csv as _csv
            from pathlib import Path as _Path

            import matplotlib.pyplot as _plt

            out_dir = _Path(getattr(self, "output_dir", "."))
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "ck_mse.csv"
            with csv_path.open("w", newline="") as fh:
                writer = _csv.writer(fh)
                writer.writerow(["tau", "mse"])
                for t, m in zip(taus, mses):
                    writer.writerow([int(t), float(m)])

            png_path = out_dir / "ck.png"
            _plt.figure()
            _plt.plot(taus, mses, marker="o")
            _plt.xlabel("Lag time τ")
            _plt.ylabel("CK MSE")
            _plt.title("CK test MSE vs τ")
            _plt.tight_layout()
            _plt.savefig(str(png_path))
            _plt.close()
            print(f"Selected τ = {int(selected)}")
        except Exception:
            pass

    def _normalize_ck_factors(self, factors: Optional[List[int]]) -> List[int]:
        if factors is None:
            return [2, 3, 4, 5]
        return [int(f) for f in factors if int(f) > 1]

    def _largest_connected_states(self, C: np.ndarray, max_states: int) -> np.ndarray:
        try:
            adj = ((C + C.T) > 0).astype(int)
            _, labels = connected_components(adj, directed=False, return_labels=True)
            counts = np.bincount(labels)
            main = int(np.argmax(counts))
            idx = np.where(labels == main)[0]
            if idx.size > max_states:
                totals = (C + C.T).sum(axis=1)
                idx = idx[np.argsort(totals[idx])[::-1]][:max_states]
            return idx
        except Exception:
            return np.arange(min(C.shape[0], max_states))

    def _count_micro_T(
        self, dtrajs: List[np.ndarray], nS: int, lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        C = np.zeros((nS, nS), dtype=float)
        for seq in dtrajs:
            seq = np.asarray(seq, dtype=int)
            if seq.size <= lag:
                continue
            for i in range(0, seq.size - lag):
                a = int(seq[i])
                b = int(seq[i + lag])
                if 0 <= a < nS and 0 <= b < nS:
                    C[a, b] += 1.0
        rows = C.sum(axis=1)
        rows[rows == 0] = 1.0
        return C / rows[:, None], C

    def _count_macro_T_and_counts(
        self, macro_trajs: List[np.ndarray], nM: int, lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        C = np.zeros((nM, nM), dtype=float)
        for seq in macro_trajs:
            if seq.size <= lag:
                continue
            for i in range(0, seq.size - lag):
                a = int(seq[i])
                b = int(seq[i + lag])
                if 0 <= a < nM and 0 <= b < nM:
                    C[a, b] += 1.0
        rows = C.sum(axis=1)
        rows[rows == 0] = 1.0
        return C / rows[:, None], C

    def _ck_mse_for_factor(
        self,
        T1: np.ndarray,
        macro_trajs: List[np.ndarray],
        nM: int,
        base_lag: int,
        factor: int,
    ) -> Tuple[Optional[float], np.ndarray]:
        try:
            T_theory = np.linalg.matrix_power(T1, int(factor))
            T_emp, C_emp = self._count_macro_T_and_counts(
                macro_trajs, nM, int(base_lag) * int(factor)
            )
            diff = T_theory - T_emp
            return float(np.mean(diff * diff)), C_emp
        except Exception:
            return None, np.zeros((nM, nM), dtype=float)

    def _micro_eigen_gap(self, k: int = 2) -> Optional[float]:
        try:
            if getattr(self, "transition_matrix", None) is not None:
                T = np.asarray(self.transition_matrix, dtype=float)
            else:
                T, _ = self._count_micro_T(
                    self.dtrajs, self.n_states, int(self.lag_time)
                )
            evals = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
            if len(evals) <= k:
                return None
            return float(evals[k - 1] - evals[k])
        except Exception:
            return None

    def _slowest_its_from_T(self, T: np.ndarray, tau: int) -> float:
        """Compute the slowest implied timescale from a transition matrix.

        Uses the second-largest eigenvalue magnitude.
        """
        try:
            evals = np.sort(np.real(np.linalg.eigvals(np.asarray(T, dtype=float))))[
                ::-1
            ]
            if evals.size < 2:
                return 0.0
            lam = float(evals[1])
            if lam <= 0 or lam >= 0.999999:
                lam = min(max(lam, 1e-12), 0.999999)
            its = -float(tau) / np.log(lam)
            if not np.isfinite(its):
                return 0.0
            return float(its)
        except Exception:
            return 0.0

    def _ck_mse_from_T(self, T1: np.ndarray, tau: int, factor: int) -> float:
        """Mean squared error between theoretical and empirical CK predictions.

        T_theory = T1^factor, T_emp estimated from data at lag tau*factor.
        """
        try:
            T_theory = np.linalg.matrix_power(np.asarray(T1, dtype=float), int(factor))
            T_emp, _ = self._count_micro_T(
                self.dtrajs, int(self.n_states), int(tau) * int(factor)
            )
            diff = T_theory - T_emp
            return float(np.mean(diff * diff))
        except Exception:
            return float("inf")
