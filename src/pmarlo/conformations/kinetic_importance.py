"""Kinetic Importance Score (KIS) calculation and validation."""

"""
The file is after the analysis. All should be working with the new test suite.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ..markov_state_model._msm_utils import ensure_connected_counts
from .results import KISResult

logger = logging.getLogger("pmarlo.conformations")


def _largest_connected_component_indices(counts: np.ndarray) -> np.ndarray:
    """Return indices for the largest connected component of a count matrix."""
    arr = np.asarray(counts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("count matrix must be square")
    if arr.size == 0:
        return np.empty((0,), dtype=int)

    adjacency = ((arr + arr.T) > 0).astype(int)
    graph = csr_matrix(adjacency)
    n_components, labels = connected_components(graph, directed=False)
    if n_components <= 1:
        return np.arange(arr.shape[0], dtype=int)

    component_sizes = np.bincount(labels, minlength=n_components)
    max_size = int(component_sizes.max())
    candidates = np.where(component_sizes == max_size)[0]

    if candidates.size == 1:
        chosen = int(candidates[0])
    else:
        flux = np.zeros(candidates.size, dtype=float)
        for idx, comp in enumerate(candidates):
            mask = labels == comp
            flux[idx] = float(np.sum(arr[np.ix_(mask, mask)]))
        chosen = int(candidates[int(np.argmax(flux))])

    return np.where(labels == chosen)[0]


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
        """Compute Kinetic Importance Scores (KIS) for all states.

        KIS is defined as

            KIS(i) = π_i × Σ_{m=1}^{k_slow} φ_{m+1}(i)^2

        where:
            - π_i is the stationary probability of state i
            - φ_{m+1}(i) is the i-th component of the (m+1)-th eigenvector
            (index 0 is the stationary eigenvector)
            - k_slow is the number of nonstationary slow eigenvectors included

        Args:
            k_slow:
                Number of slow eigenvectors to include. Can be:
                - "auto": choose automatically using select_k_slow(its=its)
                - int or str(int): explicit number of slow modes
            its:
                Implied timescales, used only when k_slow == "auto".

        Returns:
            KISResult with scores and metadata.
        """
        if self.n_states < 2:
            raise ValueError("KIS requires at least 2 states.")

        pi = np.asarray(self.pi, dtype=float)
        active_mask = pi > 0.0

        if not np.any(active_mask):
            raise ValueError("No active states with positive stationary probability.")

        # Resolve k_slow
        if k_slow == "auto":
            k_slow_val = self.select_k_slow(its=its)
        elif isinstance(k_slow, str):
            # Allow stringified integers, but fail loudly on nonsense
            try:
                k_slow_val = int(k_slow)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid k_slow={k_slow!r}. Use 'auto' or an integer."
                ) from exc
        else:
            k_slow_val = int(k_slow)

        # Validate and clamp k_slow into [1, n_states - 1]
        if k_slow_val < 1 or k_slow_val >= self.n_states:
            clamped = min(max(1, k_slow_val), self.n_states - 1)
            logger.warning(
                "k_slow=%s is out of valid range [1, %s]. "
                "Clamping to k_slow=%s.",
                k_slow_val,
                self.n_states - 1,
                clamped,
            )
            k_slow_val = clamped

        # Optionally remember for later use by other methods
        self.k_slow = k_slow_val

        logger.info("Computing KIS with k_slow=%d", k_slow_val)

        if not np.all(active_mask):
            result = self._compute_on_active_subset(
                active_mask=active_mask, k_slow=k_slow_val, its=its
            )
            self.last_result = result
            return result

        # We need the stationary eigenvector (index 0) plus k_slow_val slow modes
        n_vecs = k_slow_val + 1
        eigenvalues, eigenvectors = self._compute_eigenvectors(n_vecs)

        # Ensure we work with real parts (numerical eig can be complex)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Expect eigenvectors shape (n_vecs, n_states)
        if eigenvectors.shape != (n_vecs, self.n_states):
            raise ValueError(
                f"Expected eigenvectors with shape {(n_vecs, self.n_states)}, "
                f"got {eigenvectors.shape}."
            )

        # Build a π-orthonormal basis of the slow subspace and project onto it.
        U = self._orthonormalize_slow_subspace(eigenvectors, k_slow_val)

        # Projection norm squared in π-inner product reduces to π_i * sum_j U[i, j]^2.
        kis_scores = pi * np.sum(U**2, axis=1)

        # Rank states by KIS in descending order
        ranked_states = np.argsort(kis_scores)[::-1]

        logger.info(
            "KIS computed. Top state: %d (score: %.6f)",
            int(ranked_states[0]),
            float(kis_scores[ranked_states[0]]),
        )

        result = KISResult(
            kis_scores=kis_scores,
            k_slow=k_slow_val,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
            ranked_states=ranked_states,
        )

        # Optionally cache the last result if useful elsewhere
        self.last_result = result

        return result

    def select_k_slow(
        self,
        its: Optional[np.ndarray] = None,
        method: str = "timescale_gap",
        gap_threshold: float = 2.0,
    ) -> int:
        """Automatically select number of slow modes.

        Args:
            its: Implied timescales array. Required for 'timescale_gap' method.
            method: Selection method ('timescale_gap' or 'variance').
            gap_threshold: Minimum ratio for timescale gap in 'timescale_gap' method.

        Returns:
            Number of slow eigenvectors to include (between 2 and n_states).

        Raises:
            ValueError: If n_states < 2 or an unknown method is requested.
        """
        n_states = int(self.n_states)

        if n_states < 2:
            raise ValueError(
                "select_k_slow requires at least 2 states to define slow modes, "
                f"got n_states={n_states}"
            )

        def _clamp(k: int) -> int:
            """Clamp k into [2, n_states]."""
            k_int = int(k)
            if k_int < 2:
                k_int = 2
            if k_int > n_states:
                k_int = n_states
            return k_int

        if method == "timescale_gap":
            if its is None or len(its) < 2:
                logger.warning(
                    "Method 'timescale_gap' requested but implied timescales are "
                    "missing or too short. Falling back to default heuristic."
                )
            else:
                k_raw = self._select_by_timescale_gap(its, gap_threshold)
                k_slow = _clamp(k_raw)
                logger.debug(
                    f"Selected k_slow={k_slow} using timescale gap "
                    f"(raw={k_raw}, gap_threshold={gap_threshold})"
                )
                return k_slow

        elif method == "variance":
            k_raw = self._select_by_variance_explained()
            k_slow = _clamp(k_raw)
            logger.debug(
                f"Selected k_slow={k_slow} using variance explained (raw={k_raw})"
            )
            return k_slow

        else:
            raise ValueError(
                f"Unknown k_slow selection method '{method}'. "
                "Supported methods are 'timescale_gap' and 'variance'."
            )

        # Fallback heuristic if we could not apply the requested method
        default_k = min(5, max(2, n_states // 10))
        k_slow = _clamp(default_k)
        logger.debug(
            f"Using fallback k_slow={k_slow} "
            f"(heuristic from n_states={n_states}, raw_default={default_k})"
        )
        return k_slow

    def _select_by_timescale_gap(self, its: np.ndarray, gap_threshold: float) -> int:
        """Select k_slow based on a gap in implied timescales.

        Args:
            its: 1D array of implied timescales (one per nonstationary eigenmode),
                expected to be positive and roughly sorted from slow to fast.
            gap_threshold: Minimum acceptable ratio between consecutive timescales
                to declare a clear gap.

        Returns:
            Number of slow modes k_slow.
        """
        its = np.asarray(its, dtype=float)

        if its.ndim != 1:
            raise ValueError("its must be a 1D array of implied timescales")

        # Keep only positive, finite timescales
        valid_mask = np.isfinite(its) & (its > 0.0)
        if not np.any(valid_mask):
            logger.warning(
                "No positive finite implied timescales. "
                "Falling back to k_slow=2."
            )
            return 2

        its_valid = its[valid_mask]

        # Need at least two timescales to define a gap
        if its_valid.size < 2:
            logger.debug(
                "Only %d valid implied timescale(s). "
                "Falling back to k_slow=2.",
                its_valid.size,
            )
            return 2

        # Ensure descending order: largest timescales first (slowest modes)
        its_sorted = np.sort(its_valid)[::-1]

        # Ratios between consecutive timescales τ_i / τ_{i+1}
        denom = np.maximum(its_sorted[1:], 1e-10)
        ratios = its_sorted[:-1] / denom

        # Index of largest gap
        gap_idx = int(np.argmax(ratios))
        max_ratio = float(ratios[gap_idx])

        if max_ratio >= gap_threshold:
            # Gap between its_sorted[gap_idx] and its_sorted[gap_idx + 1]
            # Keep all modes up to the gap
            k_slow = gap_idx + 1
            logger.debug(
                "Timescale gap detected at index %d (ratio: %.2f >= %.2f). "
                "Setting k_slow=%d.",
                gap_idx,
                max_ratio,
                gap_threshold,
                k_slow,
            )
        else:
            # No clear gap. Use a small default number of slow modes.
            k_slow = min(5, its_sorted.size)
            logger.debug(
                "No clear timescale gap (max ratio: %.2f < %.2f). "
                "Using default k_slow=%d.",
                max_ratio,
                gap_threshold,
                k_slow,
            )

        # Final safety: at least 2 and at most number of valid timescales
        k_slow = max(2, min(k_slow, its_sorted.size))
        return int(k_slow)


    def _select_by_variance_explained(self, variance_threshold: float = 0.9) -> int:
        """Select k_slow based on fraction of slow spectral weight explained.

        We treat the nonstationary eigenvalues λ_k (k ≥ 2 in the usual MSM
        ordering) as defining a spectrum of "slow content" via their magnitudes
        |λ_k|. We then choose the smallest k_slow such that the cumulative sum
        over modes 2..k_slow reaches `variance_threshold` of the total weight
        over all nonstationary modes.

        This keeps:
        - consistency with the usual ordering by |λ|
        - nonnegative weights even for complex spectra
        - a monotone cumulative fraction in [0, 1]
        """

        # Handle degenerate very small MSMs explicitly
        if self.n_states <= 2:
            # With 1 or 2 states there is at most one nonstationary mode
            k_slow = min(2, self.n_states)
            logger.debug(
                "Small MSM with n_states=%d, using trivial k_slow=%d",
                self.n_states,
                k_slow,
            )
            return int(k_slow)

        # Clamp threshold to [0, 1]
        variance_threshold = float(np.clip(variance_threshold, 0.0, 1.0))

        # Compute all eigenvalues (assumed sorted by decreasing |λ| in _compute_eigenvectors)
        eigenvalues, _ = self._compute_eigenvectors(self.n_states)
        eigenvalues = np.asarray(eigenvalues)

        # Drop the stationary mode (index 0) and use magnitudes of nonstationary eigenvalues
        nonstationary = eigenvalues[1:]
        if nonstationary.size == 0:
            # Should not happen given n_states > 2, but guard anyway
            logger.debug("No nonstationary eigenvalues found, falling back to k_slow=2")
            return 2

        weights = np.abs(nonstationary)
        total = np.sum(weights)

        if not np.isfinite(total) or total <= 0.0:
            logger.debug(
                "Nonpositive or invalid total weight (total=%s), falling back to k_slow=2",
                total,
            )
            k_slow = 2
        else:
            # Cumulative fraction of slow spectral weight
            cumsum = np.cumsum(weights) / total

            # Number of nonstationary modes needed to reach threshold
            # cumsum is over modes 2,3,..., so index 0 corresponds to k=2
            insertion_point = int(
                np.searchsorted(cumsum, variance_threshold, side="right")
            )
            m_nonstat = min(nonstationary.size, insertion_point + 1)

            # Convert to k_slow index in full spectrum (include stationary mode at index 0)
            k_slow = 1 + m_nonstat

        # Enforce at least 2 modes and at most n_states
        max_k = max(2, self.n_states)
        k_slow = int(np.clip(k_slow, 2, max_k))

        logger.debug(
            "Variance-based selection (threshold=%.3f): k_slow=%d",
            variance_threshold,
            k_slow,
        )
        return int(k_slow)

    def _compute_on_active_subset(
        self,
        active_mask: np.ndarray,
        k_slow: int,
        its: Optional[np.ndarray],
    ) -> KISResult:
        """Compute KIS on the active subspace and embed back into full state space."""
        T_sub = self.T[np.ix_(active_mask, active_mask)]
        pi_sub = self.pi[active_mask]

        kis_calc = KineticImportanceScore(T_sub, pi_sub)
        sub_result = kis_calc.compute(k_slow=k_slow, its=its)

        # Embed scores back into the original state space.
        kis_scores = np.zeros(self.n_states, dtype=float)
        kis_scores[active_mask] = sub_result.kis_scores

        eigenvectors_full = np.zeros(
            (sub_result.eigenvectors.shape[0], self.n_states), dtype=float
        )
        eigenvectors_full[:, active_mask] = sub_result.eigenvectors

        ranked_states = np.argsort(kis_scores)[::-1]

        return KISResult(
            kis_scores=kis_scores,
            k_slow=sub_result.k_slow,
            eigenvectors=eigenvectors_full,
            eigenvalues=sub_result.eigenvalues,
            ranked_states=ranked_states,
            stability_metric=sub_result.stability_metric,
            bootstrap_std=sub_result.bootstrap_std,
        )

    def _compute_eigenvectors(self, n_vecs: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute leading left eigenvectors of the transition matrix.

        This assumes T is row stochastic (rows sum to 1) and acts on row
        distributions p^T via p^T T. In that convention, the stationary
        distribution and slow modes are left eigenvectors of T.

        Args:
            n_vecs: Number of eigenvectors to compute (including the stationary one).

        Returns:
            eigenvalues: Array of shape (n_vecs,) with eigenvalues of T,
                sorted by decreasing absolute value.
            eigenvectors: Array of shape (n_vecs, n_states) where each row
                is a left eigenvector of T (corresponding to the eigenvalue
                at the same index in `eigenvalues`).
        """
        if n_vecs <= 0:
            raise ValueError(f"n_vecs must be positive, got {n_vecs}")

        T = np.asarray(self.T)

        # Eigendecomposition of T^T.
        # Right eigenvectors of T^T are left eigenvectors of T.
        eigenvalues, eigenvectors_Tt = np.linalg.eig(T.T)

        # Sort by eigenvalue magnitude (slow modes have |lambda| close to 1)
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

        # Take top n_vecs
        n_vecs = min(n_vecs, len(eigenvalues))
        top_indices = sorted_indices[:n_vecs]

        eigenvalues_top = eigenvalues[top_indices]

        # np.linalg.eig returns eigenvectors as columns.
        # Select the leading ones and transpose so that each row is one eigenvector.
        eigenvectors_top = eigenvectors_Tt[:, top_indices].T  # shape: (n_vecs, n_states)

        # If eigenvectors are numerically real, drop tiny imaginary parts.
        if np.all(np.abs(np.imag(eigenvectors_top)) < 1e-10):
            eigenvectors_top = np.real(eigenvectors_top)

        return eigenvalues_top, eigenvectors_top

    def _orthonormalize_slow_subspace(
        self,
        eigenvectors: np.ndarray,
        k_slow: int,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Return a π-orthonormal basis of the slow subspace."""
        if k_slow < 1:
            raise ValueError("k_slow must be at least 1 to build a slow subspace.")

        eigenvectors = np.asarray(eigenvectors, dtype=float)
        if eigenvectors.ndim != 2 or eigenvectors.shape[0] < k_slow + 1:
            raise ValueError(
                "Eigenvectors array must contain the stationary mode and k_slow slow modes."
            )

        V = eigenvectors[1 : k_slow + 1].T  # shape (n_states, k_slow)
        pi = np.asarray(self.pi, dtype=float)
        if V.shape[0] != pi.shape[0]:
            raise ValueError(
                "Eigenvectors and stationary distribution have incompatible dimensions."
            )

        w = np.sqrt(np.maximum(pi, eps))
        W = V * w[:, None]

        Q, _ = np.linalg.qr(W, mode="reduced")
        if Q.size == 0:
            raise ValueError("Failed to obtain nontrivial slow modes during orthonormalization.")

        U = Q / w[:, None]
        return U

    def bootstrap_stability(
        self,
        dtrajs: List[np.ndarray],
        n_boot: int = 200,
        k_slow: Optional[int] = None,
        top_n: int = 10,
    ) -> Tuple[float, np.ndarray]:
            """Assess KIS ranking stability via bootstrap.

            This method uses trajectory-level bootstrap resampling to rebuild MSMs,
            recompute KIS, and quantify how stable the top states are.

            Args:
                dtrajs: List of discrete trajectories (each is a 1D array of state indices)
                n_boot: Number of bootstrap samples
                k_slow: Number of slow modes (uses self.select_k_slow() if None)
                top_n: Number of top states to track in the ranking

            Returns:
                Tuple of:
                    stability_metric: float in [0, 1]
                        Average fraction of original top_n states that remain in
                        the bootstrap top_n across successful samples.
                    bootstrap_std: np.ndarray of shape (n_states,)
                        Per state standard deviation of KIS across bootstrap samples.
            """
            if k_slow is None:
                k_slow = self.select_k_slow()

            logger.info(f"Bootstrap KIS stability with {n_boot} samples")

            # Original KIS ranking from the reference MSM
            original_result = self.compute(k_slow=k_slow)
            original_top = original_result.ranked_states[:top_n]

            # Allocate arrays for bootstrap results
            bootstrap_kis = np.full((n_boot, self.n_states), np.nan, dtype=float)
            bootstrap_rankings = np.full((n_boot, top_n), -1, dtype=int)

            rng = np.random.default_rng()
            n_trajs = len(dtrajs)

            successful = 0

            for b in range(n_boot):
                # Resample trajectories with replacement at the trajectory level
                resampled_dtrajs = [
                    dtrajs[rng.integers(0, n_trajs)] for _ in range(n_trajs)
                ]

                try:
                    # Try both possible signatures of _rebuild_msm:
                    #   _rebuild_msm(dtrajs)
                    #   _rebuild_msm(dtrajs, lag=...)
                    try:
                        T_boot, pi_boot = self._rebuild_msm(resampled_dtrajs)
                    except TypeError:
                        lag = getattr(self, "lag", 1)
                        T_boot, pi_boot = self._rebuild_msm(resampled_dtrajs, lag=lag)

                    # Compute KIS for bootstrap sample
                    kis_calc = KineticImportanceScore(T_boot, pi_boot)
                    result_boot = kis_calc.compute(k_slow=k_slow)

                    if result_boot.kis_scores.shape[0] != self.n_states:
                        raise ValueError(
                            f"Bootstrap sample {b} produced {result_boot.kis_scores.shape[0]} "
                            f"states but expected {self.n_states}"
                        )

                    bootstrap_kis[b] = result_boot.kis_scores
                    bootstrap_rankings[b] = result_boot.ranked_states[:top_n]
                    successful += 1

                except Exception as e:
                    logger.debug(f"Bootstrap sample {b} failed and is ignored: {e}")
                    continue

            if successful == 0:
                raise RuntimeError("All bootstrap samples failed, cannot assess stability")

            # Only consider rows where we have a valid ranking
            valid_mask = bootstrap_rankings[:, 0] != -1

            overlap_counts = []
            for b in range(n_boot):
                if not valid_mask[b]:
                    continue
                overlap = np.intersect1d(original_top, bootstrap_rankings[b])
                overlap_counts.append(float(len(overlap)))

            if not overlap_counts:
                raise RuntimeError("No valid bootstrap rankings collected")

            # Stability metric:
            # average fraction of original top_n states still in top_n in bootstrap
            stability_metric = float(np.mean(overlap_counts) / float(top_n))

            # Per state bootstrap standard deviation, ignoring NaNs from failed samples
            bootstrap_std = np.nanstd(bootstrap_kis, axis=0)

            logger.info(
                f"KIS stability metric: {stability_metric:.3f} "
                f"from {successful} successful bootstrap samples out of {n_boot}"
            )

            return stability_metric, bootstrap_std

    def _rebuild_msm(
    self,
    dtrajs: List[np.ndarray],
    lag: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
        """Rebuild MSM from discrete trajectories.

        Args:
            dtrajs: Discrete trajectories as lists of integer numpy arrays.
            lag: Lag time in steps. Must be a positive integer.

        Returns:
            Tuple of (T, pi), where:
                T: Full transition matrix of shape (n_states, n_states).
                pi: Full stationary distribution of shape (n_states,).

        Raises:
            ValueError: If lag is not a positive integer.
            ImportError: If deeptime is not installed.
        """
        try:
            from deeptime.markov import TransitionCountEstimator
            from deeptime.markov.msm import MaximumLikelihoodMSM
        except ImportError as exc:
            raise ImportError("MSM rebuilding requires deeptime") from exc

        # Validate and normalize lag
        if lag < 1:
            raise ValueError(f"lag must be a positive integer, got {lag}")
        lag = int(lag)

        # Count transitions at the given lag
        tce = TransitionCountEstimator(
            lagtime=lag,
            count_mode="sliding",
            sparse=False,
        )
        count_model = tce.fit(dtrajs).fetch_model()
        C = np.asarray(count_model.count_matrix, dtype=float)
        n_states = C.shape[0]

        # Restrict to largest connected set
        res = ensure_connected_counts(C)
        if res.counts.size == 0:
            # No connected subset with transitions
            # Return an identity MSM and zero stationary distribution
            T_empty = np.eye(n_states, dtype=float)
            pi_empty = np.zeros((n_states,), dtype=float)
            return T_empty, pi_empty

        active_states = np.asarray(res.active, dtype=int)
        raw_counts_active = C[np.ix_(active_states, active_states)]
        component_idx = _largest_connected_component_indices(raw_counts_active)
        if component_idx.size == 0:
            T_empty = np.eye(n_states, dtype=float)
            pi_empty = np.zeros((n_states,), dtype=float)
            return T_empty, pi_empty

        if component_idx.size != res.counts.shape[0]:
            counts_for_fit = res.counts[np.ix_(component_idx, component_idx)]
            active_states = active_states[component_idx]
        else:
            counts_for_fit = res.counts

        # Fit reversible MSM on the connected subset
        ml = MaximumLikelihoodMSM(
            lagtime=lag,
            reversible=True,
        )
        msm_model = ml.fit(counts_for_fit).fetch_model()
        T_active = np.asarray(msm_model.transition_matrix, dtype=float)
        pi_active = np.asarray(msm_model.stationary_distribution, dtype=float)

        # Embed back into the full state space
        T = np.eye(n_states, dtype=float)
        T[np.ix_(active_states, active_states)] = T_active

        pi = np.zeros((n_states,), dtype=float)
        pi[active_states] = pi_active

        return T, pi

    def hyperparameter_ensemble_stability(
        self,
        dtrajs: List[np.ndarray],
        lag_times: List[int],
        k_slow: Optional[int] = None,
    ) -> Tuple[float, np.ndarray]:
        """Assess KIS stability across an ensemble of lag times.

        This measures how robust the KIS ranking is when you rebuild the MSM
        at different lag times, while keeping the discrete state space fixed.

        Args:
            dtrajs:
                List of discrete trajectories on the same state space as the
                MSM stored in this KineticImportanceScore instance.
            lag_times:
                List of lag times to test.
            k_slow:
                Number of slow modes to include in the KIS calculation.
                If None, select_k_slow() is used.

        Returns:
            Tuple of (stability_metric, ensemble_std), where:
                stability_metric:
                    Mean overlap of the top 10 KIS states between the
                    reference MSM and the ensemble MSMs, in [0, 1].
                ensemble_std:
                    Standard deviation of KIS scores across the ensemble,
                    with shape (n_states,).
        """
        if k_slow is None:
            k_slow = self.select_k_slow()

        if len(lag_times) == 0:
            logger.warning("No lag times provided for hyperparameter ensemble")
            # Use the current MSM size for the zero vector
            original_result = self.compute(k_slow=k_slow)
            n_states = original_result.kis_scores.shape[0]
            return 0.0, np.zeros(n_states)

        logger.info("Hyperparameter ensemble over %d lag times", len(lag_times))

        # Reference KIS on the MSM stored in this instance
        original_result = self.compute(k_slow=k_slow)
        original_kis = original_result.kis_scores
        original_top10 = original_result.ranked_states[:10]
        n_states = original_kis.shape[0]

        ensemble_kis: List[np.ndarray] = []
        ensemble_rankings: List[np.ndarray] = []

        for lag in lag_times:
            try:
                # Rebuild MSM at this lag on the SAME state space
                T_new, pi_new = self._rebuild_msm(dtrajs, lag=lag)

                kis_calc = KineticImportanceScore(T_new, pi_new)
                result = kis_calc.compute(k_slow=k_slow)

                kis_vec = np.asarray(result.kis_scores)

                # Guard against shape mismatches, which would break index based comparisons
                if kis_vec.shape[0] != n_states:
                    logger.debug(
                        "Skipping ensemble member with lag=%d: "
                        "n_states mismatch (got %d, expected %d)",
                        lag,
                        kis_vec.shape[0],
                        n_states,
                    )
                    continue

                ensemble_kis.append(kis_vec)
                ensemble_rankings.append(result.ranked_states[:10])

            except Exception as e:
                logger.debug(
                    "Ensemble member with lag=%d failed: %s",
                    lag,
                    str(e),
                )

        if len(ensemble_kis) == 0:
            logger.warning("No successful ensemble members")
            return 0.0, np.zeros(n_states)

        # Compute stability: average overlap of top 10 across ensemble
        overlaps: List[float] = []
        original_top10 = np.asarray(original_top10, dtype=int)

        for ranking in ensemble_rankings:
            ranking = np.asarray(ranking, dtype=int)
            overlap_count = len(np.intersect1d(original_top10, ranking))
            overlaps.append(overlap_count / 10.0)

        stability_metric = float(np.mean(overlaps))

        # Ensemble standard deviation per state
        kis_matrix = np.stack(ensemble_kis, axis=0)  # shape: (n_ensemble, n_states)
        ensemble_std = kis_matrix.std(axis=0)

        logger.info("Hyperparameter ensemble stability: %.3f", stability_metric)

        return stability_metric, ensemble_std

    def _recluster(self, features: np.ndarray, n_clusters: int) -> List[np.ndarray]:
        """Recluster features with a different number of clusters.

        Args:
            features:
                Feature matrix with all trajectories concatenated.
                Shape: (sum_i n_frames_i, n_features)
            n_clusters:
                Number of clusters.

        Returns:
            List of discrete trajectories. Each element is a 1D array of
            cluster labels for a single original trajectory.
        """
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError as exc:
            raise ImportError("Reclustering requires scikit-learn") from exc

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        labels = kmeans.fit_predict(features)

        # This must be a list of ints with one entry per original trajectory.
        # Replace `_traj_lengths` with your actual attribute name if needed.
        traj_lengths = self._traj_lengths

        discrete_trajs: List[np.ndarray] = []
        start = 0
        for length in traj_lengths:
            end = start + length
            discrete_trajs.append(labels[start:end])
            start = end

        if start != len(labels):
            raise ValueError(
                f"Sum of trajectory lengths {start} does not match "
                f"number of labels {len(labels)}"
            )

        return discrete_trajs

    def eigenvector_subspace_overlap(
        self, T_other: np.ndarray, k: Optional[int] = None
    ) -> float:
        """Measure similarity of slow eigenspaces between two MSMs.

        This compares the k slow nonstationary eigenvectors of the reference MSM
        (self) with the k slow nonstationary eigenvectors of another MSM given by
        T_other. The comparison is done at the level of subspaces, not individual
        eigenvectors.

        Method
        ------
        1. Take the k slow nonstationary eigenvectors from each MSM.
        2. Orthonormalize these k vectors for each MSM. This gives two
        k dimensional subspaces in R^n.
        3. Compute the singular values s_i of B1.T @ B2, where B1 and B2 are
        the orthonormal bases of those subspaces.
        These singular values are cos(theta_i), where theta_i are
        the principal angles between the two subspaces.
        4. Return the mean of cos^2(theta_i) over i = 1..k.

        Interpretation
        --------------
        * overlap = 1.0  : slow subspaces are identical (up to rotation of basis)
        * overlap = 0.0  : slow subspaces are orthogonal
        * 0 < overlap < 1: partial similarity of slow dynamical modes

        Args:
            T_other: Transition matrix of the alternative MSM.
            k: Number of slow nonstationary eigenvectors to compare.
            If None, uses self.select_k_slow().

        Returns:
            A subspace overlap metric in [0.0, 1.0].
        """
        if k is None:
            k = self.select_k_slow()
        if k < 1:
            raise ValueError("k must be at least 1")

        # Eigenvectors for the reference MSM.
        # Assumed shape: evecs1 has shape (k+1, n_states)
        # row 0 is stationary, rows 1..k are slow nonstationary modes
        evals1, evecs1 = self._compute_eigenvectors(k + 1)

        # Eigenvectors for the other MSM.
        # np.linalg.eig returns eigenvectors in columns.
        evals2, evecs2_full = np.linalg.eig(T_other.T)

        # Sort eigenpairs of the other MSM by eigenvalue magnitude (slowest first).
        sorted_idx = np.argsort(np.abs(evals2))[::-1]
        evecs2_leading = evecs2_full[:, sorted_idx[: k + 1]]  # shape (n_states, k+1)

        # Take slow nonstationary modes and enforce real values.
        # V1, V2 have shape (n_states, k)
        V1 = np.real_if_close(evecs1[1 : k + 1].T)
        V2 = np.real_if_close(evecs2_leading[:, 1 : k + 1])

        # Orthonormalize columns to get bases for the two k dimensional subspaces.
        def _orthonormalize(X: np.ndarray) -> np.ndarray:
            """Return an orthonormal basis that spans the columns of X."""
            # Reduced QR gives Q with orthonormal columns spanning col(X).
            Q, _ = np.linalg.qr(X, mode="reduced")
            return Q

        B1 = _orthonormalize(V1)
        B2 = _orthonormalize(V2)

        return eigenvector_subspace_overlap_from_bases(B1, B2)


def eigenvector_subspace_overlap_from_bases(B1: np.ndarray, B2: np.ndarray) -> float:
    """Return mean cos² of principal angles between two orthonormal bases.

    Args:
        B1: Shape (n_states, k1) matrix whose columns form an orthonormal basis.
        B2: Shape (n_states, k2) matrix whose columns form an orthonormal basis.

    Returns:
        Scalar overlap in [0, 1], where 1 means identical subspaces and
        0 means orthogonal subspaces.
    """
    B1 = np.asarray(B1, dtype=float)
    B2 = np.asarray(B2, dtype=float)

    if B1.ndim != 2 or B2.ndim != 2:
        raise ValueError("Subspace bases must be provided as 2D arrays")
    if B1.shape[0] != B2.shape[0]:
        raise ValueError(
            "Subspace bases must have the same ambient dimension (same number of rows)"
        )

    # Align the number of basis vectors to the shared rank.
    m = min(B1.shape[1], B2.shape[1])
    if m == 0:
        return 0.0

    B1 = B1[:, :m]
    B2 = B2[:, :m]

    # Principal angles: singular values of B1.T @ B2 are cos(theta_i).
    M = B1.T @ B2
    s = np.linalg.svd(M, compute_uv=False)

    # Numerical guard to stay in [0, 1].
    s = np.clip(s, 0.0, 1.0)

    # Overlap is mean of cos^2(theta_i).
    return float(np.mean(s**2))
