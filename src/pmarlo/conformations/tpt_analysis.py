"""Core Transition Path Theory analysis using deeptime.

Implements TPT following the deeptime documentation exactly:
https://deeptime-ml.github.io/latest/notebooks/tpt.html

No fallbacks - raises ImportError if deeptime is not available.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from .results import TPTResult

if TYPE_CHECKING:
    from deeptime.markov.msm import MarkovStateModel as DeeptimeMSM

logger = logging.getLogger("pmarlo.conformations")


# Increased ceiling on the number of pathway iterations allowed during the
# deeptime pathway decomposition step. The deeptime implementation stops after
# ``maxiter`` iterations and emits a ``RuntimeWarning`` when it reaches that
# ceiling. Empirically, the default (1000) can be too low for larger MSMs, so we
# raise it substantially while still allowing callers to restrict the number of
# returned pathways separately.
PATHWAY_MAX_ITERATIONS = 10_000


class TPTAnalysis:
    """Transition Path Theory analysis using deeptime.

    Follows the deeptime TPT API exactly:
    https://deeptime-ml.github.io/latest/notebooks/tpt.html
    
    Uses deeptime's MarkovStateModel.reactive_flux() method directly.
    """

    def __init__(self, T: np.ndarray, pi: np.ndarray) -> None:
        """Initialize TPT analysis.

        Args:
            T: Transition matrix (n_states x n_states)
            pi: Stationary distribution (n_states,)
        
        Raises:
            ValueError: If T is not square or pi length doesn't match
        """
        self.T = np.asarray(T, dtype=float)
        self.pi = np.asarray(pi, dtype=float)
        self.n_states = T.shape[0]

        # Validate inputs
        if T.shape[0] != T.shape[1]:
            raise ValueError("Transition matrix must be square")
        if len(pi) != self.n_states:
            raise ValueError("Stationary distribution length must match T dimensions")
        
        # Create deeptime MSM object
        # This is the canonical way per deeptime docs
        from deeptime.markov.msm import MarkovStateModel
        self.msm = MarkovStateModel(self.T, stationary_distribution=self.pi)

    def analyze(
        self,
        source_states: np.ndarray,
        sink_states: np.ndarray,
        n_paths: int = 5,
        pathway_fraction: float = 0.99,
    ) -> TPTResult:
        """Run complete TPT analysis using deeptime.

        Uses msm.reactive_flux(A, B) as per deeptime documentation.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices
            n_paths: Maximum number of pathways to extract
            pathway_fraction: Fraction of flux to capture with pathways

        Returns:
            TPTResult with all analysis outputs
            
        Raises:
            ImportError: If deeptime is not available
            ValueError: If source and sink states overlap
        """
        logger.info(
            f"Running TPT analysis: {len(source_states)} source states, "
            f"{len(sink_states)} sink states"
        )

        # Ensure unique indices
        source = np.unique(source_states).astype(int).tolist()
        sink = np.unique(sink_states).astype(int).tolist()

        # Check for overlap
        if len(set(source) & set(sink)) > 0:
            raise ValueError("Source and sink states must not overlap")

        # Compute reactive flux using deeptime's canonical API
        flux = self.msm.reactive_flux(source, sink)

        # Extract committors
        q_forward = np.asarray(flux.forward_committor, dtype=float)
        q_backward = np.asarray(flux.backward_committor, dtype=float)

        # Extract fluxes
        flux_matrix = np.asarray(flux.gross_flux, dtype=float)
        net_flux = np.asarray(flux.net_flux, dtype=float)
        total_flux = float(flux.total_flux)

        # Extract rate and MFPT
        rate = float(flux.rate)
        mfpt = float(flux.mfpt)

        # Extract pathways using deeptime's pathway decomposition
        tpt_converged = True
        pathway_iterations = 0
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", category=RuntimeWarning)
            pathways, pathway_fluxes = flux.pathways(
                fraction=pathway_fraction,
                maxiter=PATHWAY_MAX_ITERATIONS,
            )
            pathway_iterations = len(pathways)

        if pathway_iterations > n_paths:
            pathways = pathways[:n_paths]
            pathway_fluxes = pathway_fluxes[:n_paths]

        for caught in caught_warnings:
            if issubclass(caught.category, RuntimeWarning) and (
                "Maximum number of iterations reached" in str(caught.message)
            ):
                tpt_converged = False
                logger.warning(
                    "TPT pathway extraction failed to converge after %d iterations (max %d): %s",
                    pathway_iterations,
                    PATHWAY_MAX_ITERATIONS,
                    caught.message,
                )
                break
            if issubclass(caught.category, RuntimeWarning):
                logger.warning(
                    "Runtime warning during TPT pathway extraction after %d iterations (max %d): %s",
                    pathway_iterations,
                    PATHWAY_MAX_ITERATIONS,
                    caught.message,
                )

        # Convert pathways to lists
        pathways = [list(map(int, path)) for path in pathways]
        pathway_fluxes = np.asarray(pathway_fluxes, dtype=float)

        # Find bottleneck states
        bottleneck_states = self.find_bottleneck_states(flux_matrix, top_n=10)

        logger.info(
            f"TPT complete: rate={rate:.3e}, MFPT={mfpt:.1f}, "
            f"total_flux={total_flux:.3e}, n_pathways={len(pathways)}"
        )

        return TPTResult(
            source_states=np.array(source, dtype=int),
            sink_states=np.array(sink, dtype=int),
            forward_committor=q_forward,
            backward_committor=q_backward,
            flux_matrix=flux_matrix,
            net_flux=net_flux,
            total_flux=total_flux,
            rate=rate,
            mfpt=mfpt,
            pathways=pathways,
            pathway_fluxes=pathway_fluxes,
            bottleneck_states=bottleneck_states,
            tpt_converged=tpt_converged,
            pathway_iterations=pathway_iterations,
            pathway_max_iterations=PATHWAY_MAX_ITERATIONS,
        )

    def compute_committor(
        self,
        source_states: np.ndarray,
        sink_states: np.ndarray,
        forward: bool = True,
    ) -> np.ndarray:
        """Compute committor probabilities using deeptime.

        The forward committor q+(i) is the probability that a trajectory
        starting from state i will reach the sink before the source.

        Uses deeptime.markov.tools.analysis.committor directly.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices
            forward: If True, compute forward committor; else backward

        Returns:
            Committor probabilities (n_states,)
            
        Raises:
            ImportError: If deeptime is not available
            ValueError: If source and sink overlap
        """
        from deeptime.markov.tools.analysis import committor

        # Ensure unique and valid indices
        source = np.unique(source_states).astype(int)
        sink = np.unique(sink_states).astype(int)

        # Check for overlap
        if len(np.intersect1d(source, sink)) > 0:
            raise ValueError("Source and sink states must not overlap")

        # Compute committor
        q = committor(self.T, source, sink, forward=forward)

        return np.asarray(q, dtype=float)

    def compute_reactive_flux_direct(
        self,
        source_states: np.ndarray,
        sink_states: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute reactive flux using deeptime's MSM.reactive_flux().

        This is the canonical way per deeptime documentation.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices

        Returns:
            Tuple of (flux_matrix, net_flux, total_flux)
            
        Raises:
            ImportError: If deeptime is not available
            ValueError: If source and sink overlap
        """
        # Ensure unique indices
        source = np.unique(source_states).astype(int).tolist()
        sink = np.unique(sink_states).astype(int).tolist()

        # Check for overlap
        if len(set(source) & set(sink)) > 0:
            raise ValueError("Source and sink states must not overlap")

        # Use deeptime's reactive_flux method
        flux = self.msm.reactive_flux(source, sink)

        flux_matrix = np.asarray(flux.gross_flux, dtype=float)
        net_flux = np.asarray(flux.net_flux, dtype=float)
        total_flux = float(flux.total_flux)

        logger.debug(f"Total reactive flux: {total_flux:.6e}")

        return flux_matrix, net_flux, total_flux

    def coarse_grain_flux(
        self,
        source_states: np.ndarray,
        sink_states: np.ndarray,
        sets: List[List[int]],
    ) -> Tuple[List[set], any]:
        """Coarse-grain reactive flux onto sets of states.

        Uses deeptime's ReactiveFlux.coarse_grain() method.

        Args:
            source_states: Source state indices  
            sink_states: Sink state indices
            sets: List of state sets for coarse-graining

        Returns:
            Tuple of (coarse_grained_sets, coarse_grained_flux)
            
        Raises:
            ImportError: If deeptime is not available
        """
        # Get reactive flux
        source = np.unique(source_states).astype(int).tolist()
        sink = np.unique(sink_states).astype(int).tolist()
        
        flux = self.msm.reactive_flux(source, sink)
        
        # Coarse-grain
        cg_sets, cg_flux = flux.coarse_grain(sets)
        
        return cg_sets, cg_flux

    def find_bottleneck_states(
        self, flux_matrix: np.ndarray, top_n: int = 10
    ) -> np.ndarray:
        """Identify states with highest reactive flux.

        Bottleneck states are critical for the transition and have high flux.

        Args:
            flux_matrix: Reactive flux matrix
            top_n: Number of top bottleneck states to return

        Returns:
            Array of state indices sorted by flux (descending)
        """
        # Sum flux passing through each state
        flux_through_state = np.sum(flux_matrix, axis=1) + np.sum(flux_matrix, axis=0)

        # Sort by flux
        sorted_indices = np.argsort(flux_through_state)[::-1]

        # Return top_n
        return sorted_indices[:top_n]

    def identify_transition_state_ensemble(
        self,
        committor: np.ndarray,
        tolerance: float = 0.1,
    ) -> np.ndarray:
        """Identify transition state ensemble from committor.

        States with q+ ≈ 0.5 are in the transition state ensemble.

        Args:
            committor: Forward committor probabilities
            tolerance: Tolerance around 0.5 (default: ±0.1)

        Returns:
            Array of state indices in the transition state ensemble
        """
        lower = 0.5 - tolerance
        upper = 0.5 + tolerance

        ts_states = np.where((committor >= lower) & (committor <= upper))[0]

        logger.info(
            f"Identified {len(ts_states)} transition state(s) "
            f"with committor in [{lower:.2f}, {upper:.2f}]"
        )

        return ts_states

    def pathway_statistics(
        self, pathways: List[List[int]], pathway_fluxes: np.ndarray
    ) -> List[dict]:
        """Compute statistics for each pathway.

        Args:
            pathways: List of pathways (each a list of state indices)
            pathway_fluxes: Flux through each pathway

        Returns:
            List of dictionaries with pathway statistics
        """
        if len(pathways) == 0:
            return []

        total_flux = np.sum(pathway_fluxes) if len(pathway_fluxes) > 0 else 1.0

        statistics = []
        for i, (path, flux) in enumerate(zip(pathways, pathway_fluxes)):
            # Compute free energy along path
            path_free_energies = []
            for state in path:
                if self.pi[state] > 0:
                    # F = -kT ln(pi), using kT = 1 for simplicity
                    fe = -np.log(self.pi[state])
                else:
                    fe = np.inf
                path_free_energies.append(fe)

            stats = {
                "pathway_id": i,
                "length": len(path),
                "flux": float(flux),
                "flux_fraction": float(flux / total_flux) if total_flux > 0 else 0.0,
                "states": path,
                "free_energies": path_free_energies,
                "max_barrier": float(np.max(path_free_energies) - path_free_energies[0])
                if len(path_free_energies) > 0
                else 0.0,
            }
            statistics.append(stats)

        return statistics

