"""State detection utilities for identifying source and sink states."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("pmarlo.conformations")


class StateDetector:
    """Detector for identifying source and sink states from MSM/FES data.

    Provides multiple methods for auto-detection and manual specification of
    source and sink states for TPT analysis.
    """

    def __init__(
        self, committor_thresholds: Tuple[float, float] = (0.05, 0.95)
    ) -> None:
        """Initialize state detector.

        Args:
            committor_thresholds: Lower and upper bounds that define the
                committor probability thresholds for classifying microstates
                as source, sink, or transition-like.
        """

        lower, upper = committor_thresholds
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("committor_thresholds must satisfy 0 ≤ lower < upper ≤ 1")
        self.committor_thresholds: Tuple[float, float] = (
            float(lower),
            float(upper),
        )

    def _resolve_metastable_count(self, value: Optional[int]) -> int:
        """Resolve the number of metastable states, with a default of 2."""
        resolved = 2 if value is None else int(value)
        if resolved < 2:
            raise ValueError("Number of metastable states must be at least 2")
        return resolved

    def auto_detect(
        self,
        T: np.ndarray,
        pi: np.ndarray,
        fes: Optional[Any] = None,
        its: Optional[np.ndarray] = None,
        n_states: Optional[int] = None,
        method: str = "auto",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Auto-detect source and sink states using multiple strategies.

        Tries methods in order: FES → timescale gap → populations

        Args:
            T: Transition matrix (n_states x n_states)
            pi: Stationary distribution (n_states,)
            fes: Free energy surface result (optional)
            its: Implied timescales array (optional)
            n_states: Number of metastable states to detect
            method: Detection method ('auto', 'fes', 'timescale', 'population')

        Returns:
            Tuple of (source_states, sink_states) as numpy arrays
        """
        target_states = self._resolve_metastable_count(n_states)

        if method == "auto":
            # Try FES first
            if fes is not None:
                try:
                    return self.detect_from_fes(fes, n_basins=target_states)
                except Exception as e:
                    logger.debug(f"FES detection failed: {e}")

            # Try timescale gap
            if its is not None:
                try:
                    return self.detect_from_timescale_gap(
                        T, pi, its, n_states=target_states
                    )
                except Exception as e:
                    logger.debug(f"Timescale gap detection failed: {e}")

            # Fall back to populations
            return self.detect_from_populations(pi, top_n=target_states)

        elif method == "fes":
            if fes is None:
                raise ValueError("FES data required for fes method")
            return self.detect_from_fes(fes, n_basins=target_states)

        elif method == "timescale":
            if its is None:
                raise ValueError("Implied timescales required for timescale method")
            return self.detect_from_timescale_gap(T, pi, its, n_states=target_states)

        elif method == "population":
            return self.detect_from_populations(pi, top_n=target_states)

        else:
            raise ValueError(
                f"Unknown detection method: {method}. "
                f"Choose from: auto, fes, timescale, population"
            )

    def detect_from_fes(
        self,
        fes: Any,
        n_basins: Optional[int] = None,
        method: str = "watershed",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect metastable basins from Free Energy Surface.

        Args:
            fes: FESResult object with F (free energy) and edges
            n_basins: Number of basins to detect
            method: Detection method ('watershed', 'local_minima', 'threshold')

        Returns:
            Tuple of (source_states, sink_states)
        """
        # Extract free energy array
        if hasattr(fes, "F"):
            F = np.asarray(fes.F)
        else:
            raise ValueError("FES object must have 'F' attribute")

        target_basins = self._resolve_metastable_count(n_basins)

        if method == "watershed":
            return self._watershed_basins(F, target_basins)
        elif method == "local_minima":
            return self._local_minima_basins(F, target_basins)
        elif method == "threshold":
            return self._threshold_basins(F, target_basins)
        else:
            raise ValueError(f"Unknown FES method: {method}")

    def _watershed_basins(
        self, F: np.ndarray, n_basins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use watershed algorithm to detect basins.

        Args:
            F: Free energy surface (2D array)
            n_basins: Number of basins

        Returns:
            Tuple of (source_states, sink_states) as state indices
        """
        try:
            from scipy.ndimage import label, minimum_filter
        except ImportError:
            raise ImportError("Watershed detection requires scipy")

        # Find local minima
        local_min = minimum_filter(F, size=3) == F
        labeled, n_labels = label(local_min)

        # Get positions and values of minima
        minima_positions = []
        minima_values = []
        for i in range(1, min(n_labels + 1, n_basins * 2)):
            coords = np.where(labeled == i)
            if len(coords[0]) > 0:
                idx = np.argmin(F[coords])
                pos = (coords[0][idx], coords[1][idx])
                minima_positions.append(pos)
                minima_values.append(F[pos])

        # Sort by free energy and take top n_basins
        sorted_indices = np.argsort(minima_values)[: min(n_basins, len(minima_values))]

        if len(sorted_indices) < 2:
            # Fall back to populations if not enough basins found
            logger.warning("Watershed found < 2 basins, using placeholder states")
            return np.array([0]), np.array([1])

        # Return first as source, last as sink
        source_idx = sorted_indices[0]
        sink_idx = sorted_indices[-1]

        # Convert 2D FES positions to 1D state indices based on minima positions
        source_state = np.ravel_multi_index(minima_positions[source_idx], F.shape)
        sink_state = np.ravel_multi_index(minima_positions[sink_idx], F.shape)

        return np.array([source_state]), np.array([sink_state])

    def _local_minima_basins(
        self, F: np.ndarray, n_basins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect basins from local minima in FES.

        Args:
            F: Free energy surface
            n_basins: Number of basins

        Returns:
            Tuple of (source_states, sink_states)
        """
        # Flatten and find n lowest values
        flat_F = F.ravel()
        sorted_indices = np.argsort(flat_F)

        # Take n_basins lowest points that are separated
        selected: List[Tuple[int, ...]] = []
        min_distance = max(2, F.shape[0] // (n_basins * 2))

        for idx in sorted_indices:
            if len(selected) >= n_basins:
                break
            # Convert to 2D position
            pos = np.unravel_index(idx, F.shape)
            # Check if far enough from already selected
            if not selected or all(
                np.linalg.norm(np.array(pos) - np.array(s)) > min_distance
                for s in selected
            ):
                selected.append(pos)

        if len(selected) < 2:
            return np.array([0]), np.array([1])

        # Map back to state indices (first basin as source, last as sink)
        source_state = np.ravel_multi_index(selected[0], F.shape)
        sink_state = np.ravel_multi_index(selected[-1], F.shape)

        return np.array([source_state]), np.array([sink_state])

    def _threshold_basins(
        self, F: np.ndarray, n_basins: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect basins using energy threshold.

        Args:
            F: Free energy surface
            n_basins: Number of basins

        Returns:
            Tuple of (source_states, sink_states)
        """
        # Find states below a threshold
        threshold = np.percentile(F, 20)  # Bottom 20% of energy
        low_energy_mask = F < threshold

        # Label connected regions
        try:
            from scipy.ndimage import label
        except ImportError:
            # Fallback without connected components
            low_indices = np.where(low_energy_mask.ravel())[0]
            if len(low_indices) < 2:
                return np.array([0]), np.array([1])
            return np.array([low_indices[0]]), np.array([low_indices[-1]])

        labeled, n_labels = label(low_energy_mask)
        if n_labels < 2:
            return np.array([0]), np.array([1])

        # Take largest regions
        region_sizes = [(i, np.sum(labeled == i)) for i in range(1, n_labels + 1)]
        region_sizes.sort(key=lambda x: x[1], reverse=True)

        # First region as source, second as sink
        source_region = region_sizes[0][0]
        sink_region = region_sizes[min(1, len(region_sizes) - 1)][0]

        source_state = np.where((labeled == source_region).ravel())[0][0]
        sink_state = np.where((labeled == sink_region).ravel())[0][0]

        return np.array([source_state]), np.array([sink_state])

    def detect_from_timescale_gap(
        self,
        T: np.ndarray,
        pi: np.ndarray,
        its: np.ndarray,
        n_states: Optional[int] = None,
        gap_threshold: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect metastable states using timescale gap and PCCA+.

        Args:
            T: Transition matrix
            pi: Stationary distribution
            its: Implied timescales array
            n_states: Number of metastable states
            gap_threshold: Minimum ratio between consecutive timescales

        Returns:
            Tuple of (source_states, sink_states)
        """
        # Find gap in timescales
        target_states = self._resolve_metastable_count(n_states)

        if len(its) < 2:
            # No gap, use populations
            return self.detect_from_populations(pi, top_n=target_states)

        # Compute ratios
        ratios = its[:-1] / np.maximum(its[1:], 1e-10)
        gap_idx = np.argmax(ratios)

        if ratios[gap_idx] < gap_threshold:
            logger.debug(
                f"No clear timescale gap found (max ratio: {ratios[gap_idx]:.2f})"
            )

        n_metastable = target_states
        if n_metastable > T.shape[0]:
            raise ValueError(
                "Requested number of metastable states exceeds the number of microstates"
            )

        # Use PCCA+ to find metastable sets
        try:
            from deeptime.markov import pcca
        except ImportError:
            raise ImportError("PCCA+ detection requires deeptime")

        model = pcca(T, n_metastable)
        memberships = np.asarray(model.memberships)
        labels = np.argmax(memberships, axis=1)

        # Find microstates in each macrostate
        macro_states = []
        macro_pops = []
        for m in range(n_metastable):
            states_in_macro = np.where(labels == m)[0]
            pop_in_macro = np.sum(pi[states_in_macro])
            macro_states.append(states_in_macro)
            macro_pops.append(pop_in_macro)

        # Sort by population and take top 2
        sorted_indices = np.argsort(macro_pops)[::-1]
        source_macro = sorted_indices[0]
        sink_macro = sorted_indices[min(1, len(sorted_indices) - 1)]

        source_states = macro_states[source_macro]
        sink_states = macro_states[sink_macro]

        return source_states, sink_states

    def detect_from_populations(
        self, pi: np.ndarray, top_n: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect states from highest populations.

        Args:
            pi: Stationary distribution
            top_n: Number of top states to consider

        Returns:
            Tuple of (source_states, sink_states)
        """
        n_states = self._resolve_metastable_count(top_n)

        # Find top populated states
        sorted_indices = np.argsort(pi)[::-1]
        top_states = sorted_indices[:n_states]

        if len(top_states) < 2:
            # Fallback
            return np.array([0]), np.array([min(1, len(pi) - 1)])

        # First as source, last as sink
        source_states = np.array([top_states[0]])
        sink_states = np.array([top_states[-1]])

        return source_states, sink_states

    def from_state_indices(
        self, source_indices: Sequence[int], sink_indices: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Specify source and sink states directly by microstate indices.

        Args:
            source_indices: List of source microstate IDs
            sink_indices: List of sink microstate IDs

        Returns:
            Tuple of (source_states, sink_states)
        """
        return np.asarray(source_indices, dtype=int), np.asarray(
            sink_indices, dtype=int
        )

    def from_cv_ranges(
        self,
        cv_data: np.ndarray,
        cv_name: str,
        source_range: Tuple[float, float],
        sink_range: Tuple[float, float],
        dtrajs: Optional[List[np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Specify source/sink states by CV value ranges.

        Args:
            cv_data: CV values for each frame (n_frames,)
            cv_name: Name of the CV (for logging)
            source_range: (min, max) tuple for source region
            sink_range: (min, max) tuple for sink region
            dtrajs: Discrete trajectories to map frames → states

        Returns:
            Tuple of (source_states, sink_states)
        """
        # Find frames in each range
        source_frames = np.where(
            (cv_data >= source_range[0]) & (cv_data <= source_range[1])
        )[0]
        sink_frames = np.where((cv_data >= sink_range[0]) & (cv_data <= sink_range[1]))[
            0
        ]

        if len(source_frames) == 0 or len(sink_frames) == 0:
            raise ValueError(
                f"No frames found in specified CV ranges for {cv_name}. "
                f"Source: {source_range}, Sink: {sink_range}"
            )

        if dtrajs is None:
            # Assume frame indices are state indices
            return source_frames, sink_frames

        # Map frames to states
        concatenated_dtrajs = np.concatenate(dtrajs)
        source_states = np.unique(concatenated_dtrajs[source_frames])
        sink_states = np.unique(concatenated_dtrajs[sink_frames])

        logger.info(
            f"CV-based detection for {cv_name}: "
            f"{len(source_states)} source states, {len(sink_states)} sink states"
        )

        return source_states, sink_states

    def classify_committor_states(
        self, committors: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classify microstates into source, sink, and transition regions."""

        values = np.asarray(committors, dtype=float)
        if values.ndim != 1:
            raise ValueError("committors array must be one-dimensional")

        lower_thresh, upper_thresh = self.committor_thresholds

        source_microstates = np.where(values <= lower_thresh)[0]
        sink_microstates = np.where(values >= upper_thresh)[0]
        transition_microstates = np.where(
            (values > lower_thresh) & (values < upper_thresh)
        )[0]

        return source_microstates, sink_microstates, transition_microstates

    def from_frame_indices(
        self,
        source_frames: Sequence[int],
        sink_frames: Sequence[int],
        dtrajs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map frame indices to state indices.

        Args:
            source_frames: Global frame indices for source
            sink_frames: Global frame indices for sink
            dtrajs: Discrete trajectories

        Returns:
            Tuple of (source_states, sink_states)
        """
        concatenated_dtrajs = np.concatenate(dtrajs)

        # Map frames to states
        source_states = np.unique(concatenated_dtrajs[list(source_frames)])
        sink_states = np.unique(concatenated_dtrajs[list(sink_frames)])

        return source_states, sink_states

    def from_macrostate_labels(
        self,
        macrostate_labels: np.ndarray,
        source_id: int,
        sink_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Specify source/sink using macrostate (PCCA+) labels.

        Args:
            macrostate_labels: Macrostate assignment for each microstate
            source_id: Macrostate ID for source
            sink_id: Macrostate ID for sink

        Returns:
            Tuple of (source_states, sink_states)
        """
        source_states = np.where(macrostate_labels == source_id)[0]
        sink_states = np.where(macrostate_labels == sink_id)[0]

        if len(source_states) == 0 or len(sink_states) == 0:
            raise ValueError(
                f"No states found for macrostate IDs {source_id} or {sink_id}"
            )

        return source_states, sink_states
