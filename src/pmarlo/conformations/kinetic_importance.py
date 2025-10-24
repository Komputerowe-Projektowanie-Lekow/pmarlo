"""Kinetic Importance Score (KIS) calculation and validation."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

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

        # Validate inputs
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
        # Determine k_slow
        if k_slow == "auto":
            k_slow_val = self.select_k_slow(its)
        else:
            k_slow_val = int(k_slow)

        # Validate k_slow
        if k_slow_val < 1 or k_slow_val >= self.n_states:
            k_slow_val = min(max(1, k_slow_val), self.n_states - 1)
            logger.warning(f"Adjusted k_slow to valid range: {k_slow_val}")

        logger.info(f"Computing KIS with k_slow={k_slow_val}")

        # Compute eigenvectors
        eigenvalues, eigenvectors = self._compute_eigenvectors(k_slow_val + 1)

        # Compute KIS scores
        # KIS(i) = π_i × Σ(φ_k(i))² for k=2 to K_slow+1 (skip first eigenvector)
        kis_scores = np.zeros(self.n_states, dtype=float)

        for i in range(self.n_states):
            # Sum squared eigenvector components for slow modes (skip k=0, the stationary)
            sum_sq = np.sum(eigenvectors[1 : k_slow_val + 1, i] ** 2)
            kis_scores[i] = self.pi[i] * sum_sq

        # Rank states by KIS
        ranked_states = np.argsort(kis_scores)[::-1]

        logger.info(
            f"KIS computed. Top state: {ranked_states[0]} "
            f"(score: {kis_scores[ranked_states[0]]:.6f})"
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
            # Default: use up to 5 slow modes or n_states/10, whichever is smaller
            default_k = min(5, max(2, self.n_states // 10))
            logger.debug(f"Using default k_slow={default_k}")
            return default_k

    def _select_by_timescale_gap(
        self, its: np.ndarray, gap_threshold: float
    ) -> int:
        """Select k_slow based on timescale gap.

        Args:
            its: Implied timescales
            gap_threshold: Minimum ratio between consecutive timescales

        Returns:
            Number of slow modes
        """
        if len(its) < 2:
            return 2

        # Compute ratios between consecutive timescales
        ratios = its[:-1] / np.maximum(its[1:], 1e-10)

        # Find largest gap
        gap_idx = np.argmax(ratios)

        if ratios[gap_idx] >= gap_threshold:
            # Use number of modes before the gap
            k_slow = gap_idx + 1
            logger.debug(
                f"Timescale gap detected at index {gap_idx} "
                f"(ratio: {ratios[gap_idx]:.2f}), k_slow={k_slow}"
            )
        else:
            # No clear gap, use default
            k_slow = min(5, len(its))
            logger.debug(f"No clear gap (max ratio: {ratios[gap_idx]:.2f}), k_slow={k_slow}")

        return max(2, k_slow)

    def _select_by_variance_explained(
        self, variance_threshold: float = 0.9
    ) -> int:
        """Select k_slow based on variance explained.

        Args:
            variance_threshold: Fraction of variance to explain

        Returns:
            Number of slow modes
        """
        # Compute all eigenvalues
        eigenvalues, _ = self._compute_eigenvectors(self.n_states)

        # Compute variance explained (using real parts of eigenvalues)
        real_eigenvalues = np.real(eigenvalues)
        sorted_eigenvalues = np.sort(real_eigenvalues)[::-1]

        # Normalize to sum to 1
        total = np.sum(sorted_eigenvalues)
        if total > 0:
            cumsum = np.cumsum(sorted_eigenvalues) / total
            # Find how many eigenvalues needed to reach threshold
            k_slow = int(np.searchsorted(cumsum, variance_threshold) + 1)
        else:
            k_slow = 2

        k_slow = max(2, min(k_slow, self.n_states - 1))
        logger.debug(f"Variance-based selection: k_slow={k_slow}")
        return k_slow

    def _compute_eigenvectors(self, n_vecs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute top eigenvectors of transition matrix.

        Args:
            n_vecs: Number of eigenvectors to compute

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        # Eigendecomposition of transpose (right eigenvectors become left)
        # We want right eigenvectors of T, which are left of T.T
        eigenvalues, eigenvectors_left = np.linalg.eig(self.T.T)

        # Sort by eigenvalue magnitude (descending)
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

        # Take top n_vecs
        n_vecs = min(n_vecs, len(eigenvalues))
        top_indices = sorted_indices[:n_vecs]

        eigenvalues_top = eigenvalues[top_indices]
        eigenvectors_top = eigenvectors_left[:, top_indices].T

        # Ensure eigenvectors are real (they should be for stochastic matrices)
        if np.all(np.abs(np.imag(eigenvectors_top)) < 1e-10):
            eigenvectors_top = np.real(eigenvectors_top)

        return eigenvalues_top, eigenvectors_top

    def bootstrap_stability(
        self,
        dtrajs: List[np.ndarray],
        n_boot: int = 200,
        k_slow: Optional[int] = None,
        top_n: int = 10,
    ) -> Tuple[float, np.ndarray]:
        """Assess KIS ranking stability via bootstrap.

        Args:
            dtrajs: Discrete trajectories
            n_boot: Number of bootstrap samples
            k_slow: Number of slow modes (uses self k_slow if None)
            top_n: Number of top states to track

        Returns:
            Tuple of (stability_metric, bootstrap_std)
        """
        if k_slow is None:
            # Use stored k_slow from compute()
            k_slow = self.select_k_slow()

        logger.info(f"Bootstrap KIS stability with {n_boot} samples")

        # Original KIS ranking
        original_result = self.compute(k_slow=k_slow)
        original_top = original_result.ranked_states[:top_n]

        # Bootstrap samples
        bootstrap_kis = np.zeros((n_boot, self.n_states))
        bootstrap_rankings = np.zeros((n_boot, top_n), dtype=int)

        rng = np.random.default_rng()

        for b in range(n_boot):
            # Resample trajectories
            resampled_dtrajs = [
                dtrajs[rng.integers(0, len(dtrajs))] for _ in range(len(dtrajs))
            ]

            # Rebuild MSM
            try:
                T_boot, pi_boot = self._rebuild_msm(resampled_dtrajs)

                # Compute KIS for bootstrap sample
                kis_calc = KineticImportanceScore(T_boot, pi_boot)
                result_boot = kis_calc.compute(k_slow=k_slow)

                bootstrap_kis[b] = result_boot.kis_scores
                bootstrap_rankings[b] = result_boot.ranked_states[:top_n]

            except Exception as e:
                logger.debug(f"Bootstrap sample {b} failed: {e}")
                bootstrap_kis[b] = original_result.kis_scores
                bootstrap_rankings[b] = original_top

        # Compute stability metric: fraction of bootstrap samples where
        # original top states remain in top_n
        overlap_counts = np.zeros(n_boot)
        for b in range(n_boot):
            overlap_counts[b] = len(np.intersect1d(original_top, bootstrap_rankings[b]))

        stability_metric = float(np.mean(overlap_counts) / top_n)

        # Compute bootstrap standard deviation
        bootstrap_std = np.std(bootstrap_kis, axis=0)

        logger.info(f"KIS stability metric: {stability_metric:.3f}")

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

        # Count transitions
        tce = TransitionCountEstimator(lagtime=lag, count_mode="sliding", sparse=False)
        count_model = tce.fit(dtrajs).fetch_model()
        C = np.asarray(count_model.count_matrix, dtype=float)
        n_states = C.shape[0]

        res = ensure_connected_counts(C)
        if res.counts.size == 0:
            T_empty = np.eye(n_states, dtype=float)
            pi_empty = np.zeros((n_states,), dtype=float)
            return T_empty, pi_empty

        ml = MaximumLikelihoodMSM(
            lagtime=int(max(1, lag)),
            reversible=True,
        )
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
            dtrajs: Discrete trajectories
            features: Feature matrix
            lag_times: List of lag times to test
            n_clusters_list: List of cluster numbers to test
            k_slow: Number of slow modes

        Returns:
            Tuple of (stability_metric, ensemble_std)
        """
        if k_slow is None:
            k_slow = self.select_k_slow()

        logger.info(
            f"Hyperparameter ensemble with {len(lag_times)} lags "
            f"and {len(n_clusters_list)} cluster sizes"
        )

        # Original ranking
        original_result = self.compute(k_slow=k_slow)
        original_top10 = original_result.ranked_states[:10]

        ensemble_kis = []
        ensemble_rankings = []

        for lag in lag_times:
            for n_clusters in n_clusters_list:
                try:
                    # Recluster and rebuild MSM
                    dtrajs_new = self._recluster(features, n_clusters)
                    T_new, pi_new = self._rebuild_msm(dtrajs_new, lag=lag)

                    # Compute KIS
                    kis_calc = KineticImportanceScore(T_new, pi_new)
                    result = kis_calc.compute(k_slow=k_slow)

                    ensemble_kis.append(result.kis_scores)
                    ensemble_rankings.append(result.ranked_states[:10])

                except Exception as e:
                    logger.debug(
                        f"Ensemble member (lag={lag}, n_clusters={n_clusters}) failed: {e}"
                    )

        if len(ensemble_kis) == 0:
            logger.warning("No successful ensemble members")
            return 0.0, np.zeros(self.n_states)

        # Compute stability: average overlap of top 10 across ensemble
        overlaps = []
        for ranking in ensemble_rankings:
            overlap = len(np.intersect1d(original_top10, ranking))
            overlaps.append(overlap / 10.0)

        stability_metric = float(np.mean(overlaps))

        # Ensemble standard deviation
        ensemble_std = np.std(ensemble_kis, axis=0)

        logger.info(f"Hyperparameter ensemble stability: {stability_metric:.3f}")

        return stability_metric, ensemble_std

    def _recluster(self, features: np.ndarray, n_clusters: int) -> List[np.ndarray]:
        """Recluster features with different number of clusters.

        Args:
            features: Feature matrix (n_frames x n_features)
            n_clusters: Number of clusters

        Returns:
            List of discrete trajectories
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise ImportError("Reclustering requires scikit-learn")

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        return [labels]

    def eigenvector_subspace_overlap(
        self, T_other: np.ndarray, k: Optional[int] = None
    ) -> float:
        """Compute subspace overlap between eigenvectors of two MSMs.

        Args:
            T_other: Alternative transition matrix
            k: Number of eigenvectors to compare (uses k_slow if None)

        Returns:
            Subspace overlap metric (0 to 1)
        """
        if k is None:
            k = self.select_k_slow()

        # Compute eigenvectors for both
        _, evecs1 = self._compute_eigenvectors(k + 1)
        _, evecs2_full = np.linalg.eig(T_other.T)

        # Sort second set by eigenvalue
        evals2, _ = np.linalg.eig(T_other.T)
        sorted_idx = np.argsort(np.abs(evals2))[::-1]
        evecs2 = evecs2_full[:, sorted_idx[: k + 1]].T

        # Compute subspace overlap using singular values
        # Take slow modes (skip first stationary)
        V1 = evecs1[1 : k + 1].T
        V2 = evecs2[1 : k + 1].T

        # Compute singular values of V1.T @ V2
        _, s, _ = np.linalg.svd(V1.T @ V2)

        # Overlap is mean of squared singular values
        overlap = float(np.mean(s**2))

        return overlap

