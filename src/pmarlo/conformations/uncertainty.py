"""Uncertainty quantification for conformations analysis."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .results import UncertaintyResult

logger = logging.getLogger("pmarlo.conformations")


class UncertaintyQuantifier:
    """Quantifier for assessing uncertainty in conformations analysis.

    Provides bootstrap and hyperparameter ensemble methods for uncertainty
    quantification of TPT observables, KIS scores, and other metrics.
    """

    def __init__(self, random_seed: Optional[int] = None) -> None:
        """Initialize uncertainty quantifier.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def bootstrap_tpt(
        self,
        dtrajs: List[np.ndarray],
        source_states: np.ndarray,
        sink_states: np.ndarray,
        n_boot: int = 200,
        lag: int = 1,
        ci_percentiles: Tuple[float, float] = (2.5, 97.5),
    ) -> Dict[str, UncertaintyResult]:
        """Bootstrap uncertainty for TPT observables.

        Args:
            dtrajs: Discrete trajectories
            source_states: Source state indices
            sink_states: Sink state indices
            n_boot: Number of bootstrap samples
            lag: Lag time for MSM
            ci_percentiles: Confidence interval percentiles

        Returns:
            Dictionary of UncertaintyResult objects for each observable
        """
        logger.info(f"Bootstrap TPT uncertainty with {n_boot} samples")

        # Storage for bootstrap samples
        boot_rates = []
        boot_mfpts = []
        boot_total_flux = []

        for b in range(n_boot):
            try:
                # Resample trajectories
                resampled_dtrajs = [
                    dtrajs[self.rng.integers(0, len(dtrajs))]
                    for _ in range(len(dtrajs))
                ]

                # Rebuild MSM
                T_boot, pi_boot = self._rebuild_msm(resampled_dtrajs, lag=lag)

                # Run TPT
                from .tpt_analysis import TPTAnalysis

                tpt = TPTAnalysis(T_boot, pi_boot)
                result = tpt.analyze(source_states, sink_states, n_paths=0)

                boot_rates.append(result.rate)
                boot_mfpts.append(result.mfpt)
                boot_total_flux.append(result.total_flux)

            except Exception as e:
                logger.debug(f"Bootstrap sample {b} failed: {e}")

        if len(boot_rates) == 0:
            logger.warning("All bootstrap samples failed")
            return {}

        # Compute statistics
        results = {}

        for name, samples in [
            ("rate", boot_rates),
            ("mfpt", boot_mfpts),
            ("total_flux", boot_total_flux),
        ]:
            samples_array = np.array(samples)
            results[name] = UncertaintyResult(
                observable_name=name,
                mean=float(np.mean(samples_array)),
                std=float(np.std(samples_array)),
                ci_lower=float(np.percentile(samples_array, ci_percentiles[0])),
                ci_upper=float(np.percentile(samples_array, ci_percentiles[1])),
                n_samples=len(samples),
                method="bootstrap",
            )

        logger.info(
            f"Bootstrap complete: rate = {results['rate'].mean:.3e} "
            f"± {results['rate'].std:.3e}"
        )

        return results

    def bootstrap_macrostate_populations(
        self,
        dtrajs: List[np.ndarray],
        n_macrostates: int,
        n_boot: int = 200,
        lag: int = 1,
        ci_percentiles: Tuple[float, float] = (2.5, 97.5),
    ) -> UncertaintyResult:
        """Bootstrap uncertainty for macrostate populations.

        Args:
            dtrajs: Discrete trajectories
            n_macrostates: Number of macrostates (PCCA+)
            n_boot: Number of bootstrap samples
            lag: Lag time
            ci_percentiles: CI percentiles

        Returns:
            UncertaintyResult for populations (array)
        """
        logger.info(f"Bootstrap macrostate populations with {n_boot} samples")

        boot_populations = []

        for b in range(n_boot):
            try:
                # Resample
                resampled_dtrajs = [
                    dtrajs[self.rng.integers(0, len(dtrajs))]
                    for _ in range(len(dtrajs))
                ]

                # Rebuild MSM
                T_boot, pi_boot = self._rebuild_msm(resampled_dtrajs, lag=lag)

                # PCCA+
                try:
                    from deeptime.markov import pcca
                except ImportError:
                    raise ImportError("PCCA+ requires deeptime")

                model = pcca(T_boot, n_macrostates)
                memberships = np.asarray(model.memberships)
                labels = np.argmax(memberships, axis=1)

                # Compute macrostate populations
                macro_pops = np.zeros(n_macrostates)
                for m in range(n_macrostates):
                    states_in_macro = np.where(labels == m)[0]
                    macro_pops[m] = np.sum(pi_boot[states_in_macro])

                boot_populations.append(macro_pops)

            except Exception as e:
                logger.debug(f"Bootstrap sample {b} failed: {e}")

        if len(boot_populations) == 0:
            return UncertaintyResult(
                observable_name="macrostate_populations",
                mean=np.zeros(n_macrostates),
                std=np.zeros(n_macrostates),
                ci_lower=np.zeros(n_macrostates),
                ci_upper=np.zeros(n_macrostates),
                n_samples=0,
                method="bootstrap",
            )

        boot_populations_array = np.array(boot_populations)

        return UncertaintyResult(
            observable_name="macrostate_populations",
            mean=np.mean(boot_populations_array, axis=0),
            std=np.std(boot_populations_array, axis=0),
            ci_lower=np.percentile(boot_populations_array, ci_percentiles[0], axis=0),
            ci_upper=np.percentile(boot_populations_array, ci_percentiles[1], axis=0),
            n_samples=len(boot_populations),
            method="bootstrap",
        )

    def bootstrap_free_energies(
        self,
        dtrajs: List[np.ndarray],
        T_K: float = 300.0,
        n_boot: int = 200,
        ci_percentiles: Tuple[float, float] = (2.5, 97.5),
    ) -> UncertaintyResult:
        """Bootstrap uncertainty for state free energies.

        Args:
            dtrajs: Discrete trajectories
            T_K: Temperature in Kelvin
            n_boot: Number of bootstrap samples
            ci_percentiles: CI percentiles

        Returns:
            UncertaintyResult for free energies (per state)
        """
        logger.info(f"Bootstrap free energies with {n_boot} samples")

        # Determine number of states
        n_states = max(max(dt) for dt in dtrajs) + 1

        boot_free_energies = []

        from scipy import constants

        kT = constants.k * T_K * constants.Avogadro / 1000.0  # kJ/mol

        for b in range(n_boot):
            try:
                # Resample
                resampled_dtrajs = [
                    dtrajs[self.rng.integers(0, len(dtrajs))]
                    for _ in range(len(dtrajs))
                ]

                # Rebuild MSM
                _, pi_boot = self._rebuild_msm(resampled_dtrajs, lag=1)

                # Compute free energies
                free_energies = -kT * np.log(np.maximum(pi_boot, 1e-10))
                boot_free_energies.append(free_energies)

            except Exception as e:
                logger.debug(f"Bootstrap sample {b} failed: {e}")

        if len(boot_free_energies) == 0:
            return UncertaintyResult(
                observable_name="free_energies",
                mean=np.zeros(n_states),
                std=np.zeros(n_states),
                ci_lower=np.zeros(n_states),
                ci_upper=np.zeros(n_states),
                n_samples=0,
                method="bootstrap",
            )

        boot_fe_array = np.array(boot_free_energies)

        return UncertaintyResult(
            observable_name="free_energies",
            mean=np.mean(boot_fe_array, axis=0),
            std=np.std(boot_fe_array, axis=0),
            ci_lower=np.percentile(boot_fe_array, ci_percentiles[0], axis=0),
            ci_upper=np.percentile(boot_fe_array, ci_percentiles[1], axis=0),
            n_samples=len(boot_free_energies),
            method="bootstrap",
        )

    def hyperparameter_ensemble(
        self,
        dtrajs: List[np.ndarray],
        features: np.ndarray,
        param_grid: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        """Build ensemble of MSMs across hyperparameter grid.

        Args:
            dtrajs: Discrete trajectories
            features: Feature matrix
            param_grid: Dictionary of parameter lists, e.g.:
                {'lag_time': [1, 2, 5], 'n_clusters': [50, 100, 150]}

        Returns:
            Dictionary of ensemble results
        """
        lag_times = param_grid.get("lag_time", [1])
        n_clusters_list = param_grid.get("n_clusters", [100])

        logger.info(
            f"Building hyperparameter ensemble: {len(lag_times)} lags × "
            f"{len(n_clusters_list)} cluster sizes"
        )

        ensemble = {"T": [], "pi": [], "dtrajs": [], "params": []}

        for lag in lag_times:
            for n_clusters in n_clusters_list:
                try:
                    # Recluster if needed
                    if n_clusters != len(np.unique(np.concatenate(dtrajs))):
                        dtrajs_new = self._recluster(features, n_clusters)
                    else:
                        dtrajs_new = dtrajs

                    # Build MSM
                    T, pi = self._rebuild_msm(dtrajs_new, lag=lag)

                    ensemble["T"].append(T)
                    ensemble["pi"].append(pi)
                    ensemble["dtrajs"].append(dtrajs_new)
                    ensemble["params"].append(
                        {"lag_time": lag, "n_clusters": n_clusters}
                    )

                except Exception as e:
                    logger.debug(
                        f"Ensemble member (lag={lag}, n={n_clusters}) failed: {e}"
                    )

        logger.info(f"Built ensemble with {len(ensemble['T'])} models")

        return ensemble

    def ensemble_observable_statistics(
        self,
        ensemble_results: List[Any],
        observable_name: str,
        ci_percentiles: Tuple[float, float] = (2.5, 97.5),
    ) -> UncertaintyResult:
        """Compute statistics for an observable across ensemble.

        Args:
            ensemble_results: List of observable values from ensemble
            observable_name: Name of the observable
            ci_percentiles: CI percentiles

        Returns:
            UncertaintyResult
        """
        if len(ensemble_results) == 0:
            return UncertaintyResult(
                observable_name=observable_name,
                mean=0.0,
                std=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                n_samples=0,
                method="hyperparameter_ensemble",
            )

        results_array = np.array(ensemble_results)

        return UncertaintyResult(
            observable_name=observable_name,
            mean=np.mean(results_array, axis=0),
            std=np.std(results_array, axis=0),
            ci_lower=np.percentile(results_array, ci_percentiles[0], axis=0),
            ci_upper=np.percentile(results_array, ci_percentiles[1], axis=0),
            n_samples=len(ensemble_results),
            method="hyperparameter_ensemble",
        )

    def convergence_diagnostics(
        self, iteration_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute convergence diagnostics from iterative results.

        Args:
            iteration_results: List of result dictionaries from iterations

        Returns:
            Dictionary of convergence metrics
        """
        if len(iteration_results) < 2:
            return {"converged": False, "reason": "insufficient_iterations"}

        # Extract timescales if available
        its_list = [r.get("its") for r in iteration_results if r.get("its") is not None]

        # Extract populations if available
        pi_list = [r.get("pi") for r in iteration_results if r.get("pi") is not None]

        diagnostics = {"n_iterations": len(iteration_results)}

        # Check ITS convergence
        if len(its_list) >= 2:
            its_changes = []
            for i in range(1, len(its_list)):
                change = np.abs(its_list[i] - its_list[i - 1])
                relative_change = change / np.maximum(its_list[i - 1], 1e-10)
                its_changes.append(np.mean(relative_change))

            diagnostics["its_convergence"] = {
                "mean_relative_change": float(np.mean(its_changes)),
                "converged": its_changes[-1] < 0.01 if its_changes else False,
            }

        # Check population convergence
        if len(pi_list) >= 2:
            pi_changes = []
            for i in range(1, len(pi_list)):
                change = np.abs(pi_list[i] - pi_list[i - 1])
                pi_changes.append(np.mean(change))

            diagnostics["population_convergence"] = {
                "mean_absolute_change": float(np.mean(pi_changes)),
                "converged": pi_changes[-1] < 0.001 if pi_changes else False,
            }

        # Overall convergence
        converged = True
        if "its_convergence" in diagnostics:
            converged = converged and diagnostics["its_convergence"]["converged"]
        if "population_convergence" in diagnostics:
            converged = converged and diagnostics["population_convergence"]["converged"]

        diagnostics["converged"] = converged

        return diagnostics

    def chapman_kolmogorov_validation(
        self,
        T: np.ndarray,
        dtrajs: List[np.ndarray],
        lag: int,
        n_macrostates: int = 3,
        test_lags: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Perform Chapman-Kolmogorov test for model validation.

        Args:
            T: Transition matrix
            dtrajs: Discrete trajectories
            lag: Base lag time
            n_macrostates: Number of metastable states for coarse-graining
            test_lags: Test lag times (multiples of base lag)

        Returns:
            Dictionary with CK test results
        """
        if test_lags is None:
            test_lags = [1, 2, 4, 8]

        logger.info(f"Chapman-Kolmogorov test at lags: {test_lags}")

        try:
            from deeptime.markov import pcca
        except ImportError:
            raise ImportError("CK test requires deeptime")

        # Coarse-grain to macrostates
        try:
            model = pcca(T, n_macrostates)
            memberships = np.asarray(model.memberships)
            labels = np.argmax(memberships, axis=1)
        except Exception as e:
            logger.warning(f"PCCA+ failed: {e}")
            return {"success": False, "error": str(e)}

        # Map dtrajs to macrostates
        macro_dtrajs = [labels[dtraj] for dtraj in dtrajs]

        # Compute predicted and estimated macrostate transitions
        results = []

        for test_lag in test_lags:
            actual_lag = lag * test_lag

            try:
                # Estimate at test lag
                from deeptime.markov import TransitionCountEstimator

                tce = TransitionCountEstimator(
                    lagtime=actual_lag, count_mode="sliding", sparse=False
                )
                count_model = tce.fit(macro_dtrajs).fetch_model()
                C_test = np.asarray(count_model.count_matrix, dtype=float)

                # Normalize
                row_sums = C_test.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                T_estimated = C_test / row_sums

                # Predict from original model
                # Coarse-grain original T
                T_macro = self._coarse_grain_T(T, labels, n_macrostates)
                T_predicted = np.linalg.matrix_power(T_macro, test_lag)

                # Compute error
                error = np.linalg.norm(T_estimated - T_predicted, ord="fro")

                results.append(
                    {
                        "test_lag": test_lag,
                        "error": float(error),
                        "T_estimated": T_estimated.tolist(),
                        "T_predicted": T_predicted.tolist(),
                    }
                )

            except Exception as e:
                logger.debug(f"CK test at lag {test_lag} failed: {e}")

        return {"success": True, "results": results, "n_macrostates": n_macrostates}

    def _rebuild_msm(
        self, dtrajs: List[np.ndarray], lag: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rebuild MSM from discrete trajectories."""
        try:
            from deeptime.markov import TransitionCountEstimator
        except ImportError:
            raise ImportError("MSM rebuilding requires deeptime")

        tce = TransitionCountEstimator(lagtime=lag, count_mode="sliding", sparse=False)
        count_model = tce.fit(dtrajs).fetch_model()
        C = np.asarray(count_model.count_matrix, dtype=float)

        # Normalize
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        T = C / row_sums

        # Stationary distribution
        eigenvalues, eigenvectors = np.linalg.eig(T.T)
        stationary_idx = np.argmax(np.abs(eigenvalues))
        pi = np.real(eigenvectors[:, stationary_idx])
        pi = pi / np.sum(pi)

        return T, pi

    def _recluster(self, features: np.ndarray, n_clusters: int) -> List[np.ndarray]:
        """Recluster features."""
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            raise ImportError("Reclustering requires scikit-learn")

        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        return [labels]

    def _coarse_grain_T(
        self, T: np.ndarray, labels: np.ndarray, n_macrostates: int
    ) -> np.ndarray:
        """Coarse-grain transition matrix to macrostates."""
        T_macro = np.zeros((n_macrostates, n_macrostates))

        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                macro_i = labels[i]
                macro_j = labels[j]
                T_macro[macro_i, macro_j] += T[i, j]

        # Normalize
        row_sums = T_macro.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        T_macro = T_macro / row_sums

        return T_macro
