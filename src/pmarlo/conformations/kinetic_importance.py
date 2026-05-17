"""Kinetic Importance Score (KIS) calculation and validation."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import eigs as sparse_eigs

from ..markov_state_model._msm_utils import ensure_connected_counts
from .results import KISResult

logger = logging.getLogger("pmarlo.conformations")


class KineticImportanceScore:
    """Calculator for Kinetic Importance Score (KIS).

    The KIS metric identifies states that are both thermodynamically populated
    and kinetically important for the system's slow dynamics.

    Formula: KIS(i) = π_i × Σ(φ_k(i))² for k=2 to K_slow

    where:
    - π_i is the stationary probability of state i
    - φ_k(i) is the i-th component of the k-th eigenvector
    - K_slow is the number of slow modes to include
    """

    def __init__(self, T: np.ndarray, pi: np.ndarray) -> None:
        """Initialize KIS calculator.

        Args:
            T: Transition matrix (n_states x n_states)
            pi: Stationary distribution (n_states,)
        """
        self.T = np.asarray(T, dtype=float)
        self.pi = np.asarray(pi, dtype=float)
        self.n_states = T.shape[0]

        if T.shape[0] != T.shape[1]:
            raise ValueError("Transition matrix must be square")
        if len(pi) != self.n_states:
            raise ValueError("Stationary distribution length must match T dimensions")

    def compute(
        self,
        k_slow: int | str = "auto",
        its: Optional[np.ndarray] = None,
    ) -> KISResult:
        """Compute KIS scores for all states.

        Args:
            k_slow: Number of slow eigenvectors to include ('auto' or integer)
            its: Implied timescales (for auto k_slow selection)

        Returns:
            KISResult with scores and metadata
        """
        if k_slow == "auto":
            k_slow_val = self.select_k_slow(its)
        else:
            k_slow_val = int(k_slow)

        if k_slow_val < 1 or k_slow_val >= self.n_states:
            raise ValueError(
                f"k_slow must be between 1 and n_states-1 ({self.n_states - 1}), "
                f"got {k_slow_val}"
            )

        logger.info("Computing KIS with k_slow=%d", k_slow_val)

        eigenvalues, eigenvectors = self._compute_eigenvectors(k_slow_val + 1)

        # KIS(i) = π_i × Σ(φ_k(i))² for k=2 to K_slow+1 (skip first / stationary eigenvector)
        kis_scores = np.zeros(self.n_states, dtype=float)
        for i in range(self.n_states):
            sum_sq = np.sum(eigenvectors[1 : k_slow_val + 1, i] ** 2)
            kis_scores[i] = self.pi[i] * sum_sq

        ranked_states = np.argsort(kis_scores)[::-1]

        logger.info(
            "KIS computed. Top state: %d (score: %.6f)",
            ranked_states[0],
            kis_scores[ranked_states[0]],
        )

        return KISResult(
            kis_scores=kis_scores,
            k_slow=k_slow_val,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            ranked_states=ranked_states,
        )

    def select_k_slow(
        self,
        its: Optional[np.ndarray] = None,
        method: str = "timescale_gap",
        gap_threshold: float = 2.0,
    ) -> int:
        """Automatically select number of slow modes.

        Args:
            its: Implied timescales array
            method: Selection method ('timescale_gap' or 'variance')
            gap_threshold: Minimum ratio for timescale gap

        Returns:
            Number of slow eigenvectors to include
        """
        if method == "timescale_gap" and its is not None:
            return self._select_by_timescale_gap(its, gap_threshold)
        elif method == "variance":
            return self._select_by_variance_explained()
        else:
            default_k = min(5, max(2, self.n_states // 10))
            logger.debug("Using default k_slow=%d", default_k)
            return int(default_k)

    def _select_by_timescale_gap(self, its: np.ndarray, gap_threshold: float) -> int:
        """Select k_slow based on timescale gap."""
        if len(its) < 2:
            return 2

        ratios = its[:-1] / np.maximum(its[1:], 1e-10)
        gap_idx = int(np.argmax(ratios))

        if ratios[gap_idx] >= gap_threshold:
            k_slow = gap_idx + 1
            logger.debug(
                "Timescale gap detected at index %d (ratio: %.2f), k_slow=%d",
                gap_idx,
                ratios[gap_idx],
                k_slow,
            )
        else:
            k_slow = int(min(5, len(its)))
            logger.debug(
                "No clear gap (max ratio: %.2f), k_slow=%d", ratios[gap_idx], k_slow
            )

        return int(max(2, k_slow))

    def _select_by_variance_explained(self, variance_threshold: float = 0.9) -> int:
        """Select k_slow based on variance explained by squared eigenvalues (analogous to PCA)."""
        eigenvalues, _ = self._compute_eigenvectors(self.n_states)

        # Use squared real eigenvalues — variance contribution is proportional to λ²
        sq_eigenvalues = np.real(eigenvalues) ** 2
        sorted_sq = np.sort(sq_eigenvalues)[::-1]

        total = np.sum(sorted_sq)
        if total > 0:
            cumsum = np.cumsum(sorted_sq) / total
            k_slow = int(np.searchsorted(cumsum, variance_threshold) + 1)
        else:
            k_slow = 2

        k_slow = int(max(2, min(k_slow, self.n_states - 1)))
        logger.debug("Variance-based selection: k_slow=%d", k_slow)
        return k_slow

    @staticmethod
    def _compute_eigenvectors_for(
        T: np.ndarray, n_vecs: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute top eigenvectors of a transition matrix.

        Uses a partial sparse eigensolver when possible to avoid full decomposition.

        Args:
            T: Transition matrix
            n_vecs: Number of eigenvectors to compute

        Returns:
            Tuple of (eigenvalues, eigenvectors) where eigenvectors[k] is the k-th row.
        """
        n = T.shape[0]
        n_vecs = min(n_vecs, n)

        # scipy.sparse.linalg.eigs requires k < n - 1; fall back for small/full requests
        if n_vecs < n - 1:
            eigenvalues, evecs_left = sparse_eigs(T.T, k=n_vecs, which="LM")
            order = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors_top = evecs_left[:, order].T
        else:
            all_evals, all_evecs = np.linalg.eig(T.T)
            order = np.argsort(np.abs(all_evals))[::-1][:n_vecs]
            eigenvalues = all_evals[order]
            eigenvectors_top = all_evecs[:, order].T

        imag_magnitude = float(np.abs(np.imag(eigenvectors_top)).max())
        if imag_magnitude > 1e-6:
            logger.warning(
                "Eigenvectors have non-negligible imaginary components "
                "(max=%.2e). Taking real part anyway.",
                imag_magnitude,
            )
        return np.real(eigenvalues), np.real(eigenvectors_top)

    def _compute_eigenvectors(self, n_vecs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute top eigenvectors of self.T."""
        return self._compute_eigenvectors_for(self.T, n_vecs)

    def bootstrap_stability(
        self,
        dtrajs: List[np.ndarray],
        n_boot: int = 200,
        k_slow: Optional[int] = None,
        top_n: int = 10,
        random_seed: Optional[int] = None,
        lag: int = 1,
    ) -> Tuple[float, np.ndarray]:
        """Assess KIS ranking stability via bootstrap.

        Args:
            dtrajs: Discrete trajectories
            n_boot: Number of bootstrap samples
            k_slow: Number of slow modes (uses select_k_slow if None)
            top_n: Number of top states to track
            random_seed: Seed for reproducibility
            lag: Lag time used when rebuilding bootstrap MSMs

        Returns:
            Tuple of (stability_metric, bootstrap_std)
        """
        if k_slow is None:
            k_slow = self.select_k_slow()

        logger.info("Bootstrap KIS stability with %d samples", n_boot)

        original_result = self.compute(k_slow=k_slow)
        original_top = original_result.ranked_states[:top_n]

        rng = np.random.default_rng(random_seed)

        valid_kis: List[np.ndarray] = []
        valid_rankings: List[np.ndarray] = []
        failed_count = 0

        for b in range(n_boot):
            resampled_dtrajs = [
                dtrajs[rng.integers(0, len(dtrajs))] for _ in range(len(dtrajs))
            ]
            try:
                T_boot, pi_boot = self._rebuild_msm(resampled_dtrajs, lag=lag)
                kis_calc = KineticImportanceScore(T_boot, pi_boot)
                result_boot = kis_calc.compute(k_slow=k_slow)
                valid_kis.append(result_boot.kis_scores)
                valid_rankings.append(result_boot.ranked_states[:top_n])
            except Exception as exc:
                failed_count += 1
                logger.debug("Bootstrap sample %d failed: %s", b, exc)

        if failed_count > 0:
            logger.warning(
                "%d/%d bootstrap samples failed and were excluded", failed_count, n_boot
            )
        if len(valid_kis) == 0:
            raise RuntimeError("All bootstrap samples failed; cannot compute stability")

        bootstrap_rankings = np.array(valid_rankings, dtype=int)
        overlap_counts = np.array(
            [len(np.intersect1d(original_top, row)) for row in bootstrap_rankings],
            dtype=float,
        )
        stability_metric = float(np.mean(overlap_counts) / top_n)
        bootstrap_std = np.std(np.array(valid_kis), axis=0)

        logger.info("KIS stability metric: %.3f", stability_metric)

        return stability_metric, bootstrap_std

    def _rebuild_msm(
        self, dtrajs: List[np.ndarray], lag: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rebuild MSM from discrete trajectories.

        Args:
            dtrajs: Discrete trajectories
            lag: Lag time

        Returns:
            Tuple of (T, pi)
        """
        try:
            from deeptime.markov import TransitionCountEstimator
            from deeptime.markov.msm import MaximumLikelihoodMSM
        except ImportError as exc:
            raise ImportError("MSM rebuilding requires deeptime") from exc

        tce = TransitionCountEstimator(lagtime=lag, count_mode="sliding", sparse=False)
        count_model = tce.fit(dtrajs).fetch_model()
        C = np.asarray(count_model.count_matrix, dtype=float)
        n_states = C.shape[0]

        res = ensure_connected_counts(C)
        if res.counts.size == 0:
            return np.eye(n_states, dtype=float), np.zeros((n_states,), dtype=float)

        ml = MaximumLikelihoodMSM(lagtime=int(max(1, lag)), reversible=True)
        msm_model = ml.fit(res.counts).fetch_model()
        T_active = np.asarray(msm_model.transition_matrix, dtype=float)
        pi_active = np.asarray(msm_model.stationary_distribution, dtype=float)

        T = np.eye(n_states, dtype=float)
        T[np.ix_(res.active, res.active)] = T_active
        pi = np.zeros((n_states,), dtype=float)
        pi[res.active] = pi_active

        return T, pi

    def hyperparameter_ensemble_stability(
        self,
        dtrajs: List[np.ndarray],
        features: np.ndarray,
        lag_times: List[int],
        n_clusters_list: List[int],
        k_slow: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
        """Assess KIS stability across hyperparameter ensemble.

        Args:
            dtrajs: Discrete trajectories (used to infer trajectory lengths for reclustering)
            features: Feature matrix (n_frames x n_features)
            lag_times: List of lag times to test
            n_clusters_list: List of cluster numbers to test
            k_slow: Number of slow modes

        Returns:
            Tuple of (stability_metric, ensemble_std)
        """
        if k_slow is None:
            k_slow = self.select_k_slow()

        logger.info(
            "Hyperparameter ensemble with %d lags and %d cluster sizes",
            len(lag_times),
            len(n_clusters_list),
        )

        original_result = self.compute(k_slow=k_slow)
        original_top10 = original_result.ranked_states[:10]
        traj_lengths = [len(d) for d in dtrajs]

        ensemble_kis: List[np.ndarray] = []
        ensemble_rankings: List[np.ndarray] = []
        failed_count = 0

        for lag in lag_times:
            for n_clusters in n_clusters_list:
                try:
                    dtrajs_new = self._recluster(
                        features, n_clusters, traj_lengths=traj_lengths
                    )
                    T_new, pi_new = self._rebuild_msm(dtrajs_new, lag=lag)
                    kis_calc = KineticImportanceScore(T_new, pi_new)
                    result = kis_calc.compute(k_slow=k_slow)
                    ensemble_kis.append(result.kis_scores)
                    ensemble_rankings.append(result.ranked_states[:10])
                except Exception as exc:
                    failed_count += 1
                    logger.debug(
                        "Ensemble member (lag=%d, n_clusters=%d) failed: %s",
                        lag,
                        n_clusters,
                        exc,
                    )

        if failed_count > 0:
            logger.warning(
                "%d/%d ensemble members failed and were excluded",
                failed_count,
                len(lag_times) * len(n_clusters_list),
            )
        if len(ensemble_kis) == 0:
            raise RuntimeError(
                "All hyperparameter ensemble members failed; cannot compute stability"
            )

        overlaps = [
            len(np.intersect1d(original_top10, ranking)) / 10.0
            for ranking in ensemble_rankings
        ]
        stability_metric = float(np.mean(overlaps))
        ensemble_std = np.std(np.array(ensemble_kis), axis=0)

        logger.info("Hyperparameter ensemble stability: %.3f", stability_metric)

        return stability_metric, ensemble_std

    def _recluster(
        self,
        features: np.ndarray,
        n_clusters: int,
        traj_lengths: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """Recluster features with a different number of clusters.

        Args:
            features: Feature matrix (n_frames x n_features)
            n_clusters: Number of clusters
            traj_lengths: Per-trajectory frame counts; when provided the flat label
                array is split back into per-trajectory discrete trajectories.

        Returns:
            List of discrete trajectories
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise ImportError("Reclustering requires scikit-learn")

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        if traj_lengths is None:
            return [labels]

        dtrajs: List[np.ndarray] = []
        offset = 0
        for length in traj_lengths:
            dtrajs.append(labels[offset : offset + length])
            offset += length
        return dtrajs

    def eigenvector_subspace_overlap(
        self, T_other: np.ndarray, k: Optional[int] = None
    ) -> float:
        """Compute subspace overlap between eigenvectors of two MSMs.

        Args:
            T_other: Alternative transition matrix
            k: Number of eigenvectors to compare (uses select_k_slow if None)

        Returns:
            Subspace overlap metric (0 to 1)
        """
        if k is None:
            k = self.select_k_slow()

        _, evecs1 = self._compute_eigenvectors(k + 1)
        _, evecs2 = self._compute_eigenvectors_for(
            np.asarray(T_other, dtype=float), k + 1
        )

        # Skip the first (stationary) eigenvector; compare slow subspaces
        V1 = evecs1[1 : k + 1].T
        V2 = evecs2[1 : k + 1].T

        _, s, _ = np.linalg.svd(V1.T @ V2)
        return float(np.mean(s**2))
