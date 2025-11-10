"""Transition Path Theory (TPT) mixin for EnhancedMSM.

Provides direct TPT functionality on MSM objects, delegating to deeptime's
MarkovStateModel.reactive_flux() API exactly as documented in:
https://deeptime-ml.github.io/latest/notebooks/tpt.html
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Tuple

import numpy as np

if TYPE_CHECKING:
    from deeptime.markov.reactive_flux import ReactiveFlux

logger = logging.getLogger("pmarlo.markov_state_model")


class _HasTPTAttrs(Protocol):
    """Protocol for attributes required by TPTMixin."""

    transition_matrix: Optional[np.ndarray]
    stationary_distribution: Optional[np.ndarray]
    n_states: int


class TPTMixin:
    """Mixin providing Transition Path Theory analysis using deeptime.

    This mixin adds TPT methods directly to the MSM class, following
    deeptime's API exactly. It wraps deeptime's MarkovStateModel.reactive_flux()
    method and provides convenient access to TPT quantities.

    All methods delegate to deeptime - no custom implementations.
    """

    def reactive_flux(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
    ) -> ReactiveFlux:
        """Compute reactive flux using deeptime's MarkovStateModel.reactive_flux().

        This is the canonical TPT method that computes:
        - Forward and backward committors
        - Gross flux matrix: f_ij = q_i^- * π_i * T_ij * q_j^+
        - Net flux matrix: f_ij^+ = max(0, f_ij - f_ji)
        - Total reactive flux from source to sink
        - Transition rate and mean first passage time

        Args:
            source_states: Source state indices (A)
            sink_states: Sink state indices (B)

        Returns:
            ReactiveFlux object from deeptime with all TPT quantities

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built or states overlap

        Example:
            >>> msm = EnhancedMSM(...)
            >>> msm.build_msm(lag_time=20)
            >>> flux = msm.reactive_flux([0, 1], [8, 9])
            >>> print(f"Rate: {flux.rate:.3e}")
            >>> print(f"MFPT: {flux.mfpt:.1f}")
        """
        if self.transition_matrix is None or self.stationary_distribution is None:
            raise ValueError(
                "Must call build_msm() before computing reactive flux. "
                "Transition matrix and stationary distribution are required."
            )

        from deeptime.markov.msm import MarkovStateModel

        # Convert to lists for deeptime API
        source = np.unique(np.asarray(source_states, dtype=int)).tolist()
        sink = np.unique(np.asarray(sink_states, dtype=int)).tolist()

        # Check for overlap
        if len(set(source) & set(sink)) > 0:
            raise ValueError(
                "Source and sink states must not overlap. "
                f"Source: {source}, Sink: {sink}"
            )

        # Create deeptime MSM and compute reactive flux
        msm = MarkovStateModel(
            self.transition_matrix, stationary_distribution=self.stationary_distribution
        )

        logger.info(
            f"Computing reactive flux: {len(source)} source states -> "
            f"{len(sink)} sink states"
        )

        flux = msm.reactive_flux(source, sink)

        logger.info(
            f"Reactive flux computed: rate={flux.rate:.3e}, "
            f"MFPT={flux.mfpt:.1f}, total_flux={flux.total_flux:.3e}"
        )

        return flux

    def compute_committor(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
        forward: bool = True,
    ) -> np.ndarray:
        """Compute committor probabilities using deeptime.

        The forward committor q+(i) is the probability that a trajectory
        starting from state i reaches the sink before the source.

        Uses deeptime.markov.tools.analysis.committor directly.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices
            forward: If True, compute forward committor q+; else backward q-

        Returns:
            Committor probabilities for all states (n_states,)

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built or states overlap

        Example:
            >>> msm = EnhancedMSM(...)
            >>> msm.build_msm(lag_time=20)
            >>> q_forward = msm.compute_committor([0, 1], [8, 9])
            >>> # States with q+ ≈ 0.5 are in the transition state ensemble
            >>> ts_ensemble = np.where((q_forward > 0.45) & (q_forward < 0.55))[0]
        """
        if self.transition_matrix is None:
            raise ValueError("Must call build_msm() before computing committors")

        from deeptime.markov.tools.analysis import committor

        source = np.unique(np.asarray(source_states, dtype=int))
        sink = np.unique(np.asarray(sink_states, dtype=int))

        # Check for overlap
        if len(np.intersect1d(source, sink)) > 0:
            raise ValueError("Source and sink states must not overlap")

        q = committor(self.transition_matrix, source, sink, forward=forward)

        logger.info(
            f"Computed {'forward' if forward else 'backward'} committor: "
            f"min={np.min(q):.3f}, max={np.max(q):.3f}"
        )

        return np.asarray(q, dtype=float)

    def pathway_decomposition(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
        fraction: float = 0.99,
        maxiter: int = 10000,
    ) -> Tuple[List[List[int]], np.ndarray]:
        """Decompose reactive flux into dominant pathways.

        Uses deeptime's ReactiveFlux.pathways() method to iteratively
        extract pathways by removing flux until the specified fraction
        of total flux is captured.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices
            fraction: Fraction of total flux to capture (default 0.99)
            maxiter: Maximum number of pathways to extract

        Returns:
            Tuple of (pathways, pathway_fluxes) where:
            - pathways: List of pathways, each a list of state indices
            - pathway_fluxes: Array of flux through each pathway

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built

        Example:
            >>> msm = EnhancedMSM(...)
            >>> msm.build_msm(lag_time=20)
            >>> paths, fluxes = msm.pathway_decomposition([0], [9], fraction=0.95)
            >>> for i, (path, flux) in enumerate(zip(paths, fluxes)):
            ...     print(f"Pathway {i}: {path}, flux={flux:.3e}")
        """
        flux = self.reactive_flux(source_states, sink_states)

        logger.info(
            f"Extracting pathways: fraction={fraction}, maxiter={maxiter}"
        )

        pathways, pathway_fluxes = flux.pathways(fraction=fraction, maxiter=maxiter)

        # Convert to standard Python types
        pathways = [list(map(int, path)) for path in pathways]
        pathway_fluxes = np.asarray(pathway_fluxes, dtype=float)

        logger.info(f"Extracted {len(pathways)} pathways capturing {fraction*100}% of flux")

        return pathways, pathway_fluxes

    def coarse_grain_flux(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
        sets: List[List[int]],
    ) -> Tuple[List[set[int]], Any]:
        """Coarse-grain reactive flux onto sets of states.

        Uses deeptime's ReactiveFlux.coarse_grain() method to aggregate
        flux onto user-defined sets of microstates.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices
            sets: List of state sets for coarse-graining

        Returns:
            Tuple of (coarse_grained_sets, coarse_grained_flux) where
            coarse_grained_flux has attributes like gross_flux, net_flux, etc.

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built

        Example:
            >>> msm = EnhancedMSM(...)
            >>> msm.build_msm(lag_time=20)
            >>> # Define macrostates
            >>> sets = [[0, 1, 2], [3, 4], [7, 8, 9]]
            >>> cg_sets, cg_flux = msm.coarse_grain_flux([0, 1], [8, 9], sets)
            >>> print(cg_flux.gross_flux)  # 3x3 matrix for the 3 macrostates
        """
        flux = self.reactive_flux(source_states, sink_states)

        logger.info(f"Coarse-graining flux onto {len(sets)} sets")

        cg_sets, cg_flux = flux.coarse_grain(sets)

        logger.info("Coarse-graining complete")

        return cg_sets, cg_flux

    def get_net_flux(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
    ) -> np.ndarray:
        """Get net reactive flux matrix: f_ij^+ = max(0, f_ij - f_ji).

        The net flux removes cycles and shows the directional flow
        from source to sink.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices

        Returns:
            Net flux matrix (n_states x n_states)

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built
        """
        flux = self.reactive_flux(source_states, sink_states)
        return np.asarray(flux.net_flux, dtype=float)

    def get_gross_flux(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
    ) -> np.ndarray:
        """Get gross reactive flux matrix: f_ij = q_i^- * π_i * T_ij * q_j^+.

        The gross flux includes cycles and represents the total flow
        between states.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices

        Returns:
            Gross flux matrix (n_states x n_states)

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built
        """
        flux = self.reactive_flux(source_states, sink_states)
        return np.asarray(flux.gross_flux, dtype=float)

    def get_transition_rate(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
    ) -> float:
        """Get transition rate k_AB from source to sink.

        The rate is k_AB = total_flux / (sum_i π_i * q_i^-)

        Args:
            source_states: Source state indices
            sink_states: Sink state indices

        Returns:
            Transition rate (1/time_units)

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built
        """
        flux = self.reactive_flux(source_states, sink_states)
        return float(flux.rate)

    def get_mfpt(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
    ) -> float:
        """Get mean first passage time (MFPT) from source to sink.

        The MFPT is 1/k_AB where k_AB is the transition rate.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices

        Returns:
            Mean first passage time (time_units)

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built
        """
        flux = self.reactive_flux(source_states, sink_states)
        return float(flux.mfpt)

    def identify_transition_state_ensemble(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
        tolerance: float = 0.1,
    ) -> np.ndarray:
        """Identify transition state ensemble from committor probabilities.

        States with forward committor q+ ≈ 0.5 are in the transition
        state ensemble, as they have equal probability of reaching
        either source or sink.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices
            tolerance: Tolerance around 0.5 (default ±0.1)

        Returns:
            Array of state indices in the transition state ensemble

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built
        """
        q_forward = self.compute_committor(source_states, sink_states, forward=True)

        lower = 0.5 - tolerance
        upper = 0.5 + tolerance

        ts_states = np.where((q_forward >= lower) & (q_forward <= upper))[0]

        logger.info(
            f"Identified {len(ts_states)} transition state(s) "
            f"with committor in [{lower:.2f}, {upper:.2f}]"
        )

        return ts_states

    def find_bottleneck_states(
        self: _HasTPTAttrs,
        source_states: List[int] | np.ndarray,
        sink_states: List[int] | np.ndarray,
        top_n: int = 10,
    ) -> np.ndarray:
        """Identify bottleneck states with highest reactive flux.

        Bottleneck states are critical for the A→B transition as
        the reactive flux passes through them.

        Args:
            source_states: Source state indices
            sink_states: Sink state indices
            top_n: Number of top bottleneck states to return

        Returns:
            Array of state indices sorted by flux (descending)

        Raises:
            ImportError: If deeptime is not installed
            ValueError: If transition matrix not built
        """
        flux_matrix = self.get_gross_flux(source_states, sink_states)

        # Sum flux passing through each state
        flux_through_state = 0.5 * (
            np.sum(flux_matrix, axis=1) + np.sum(flux_matrix, axis=0)
        )

        # Sort by flux
        sorted_indices = np.argsort(flux_through_state)[::-1]

        logger.info(
            f"Identified top {top_n} bottleneck states with flux "
            f"range [{flux_through_state[sorted_indices[0]]:.3e}, "
            f"{flux_through_state[sorted_indices[min(top_n-1, len(sorted_indices)-1)]]:.3e}]"
        )

        return sorted_indices[:top_n]


__all__ = ["TPTMixin"]

