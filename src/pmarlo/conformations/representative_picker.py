"""Representative structure selection with corrected weighting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("pmarlo.conformations")


@dataclass(frozen=True)
class FrameIndexLookup:
    """Lookup table that maps global frames to trajectory-local indices."""

    state_by_global_frame: np.ndarray
    trajectory_index: np.ndarray
    local_frame_index: np.ndarray

    def frames_for_state(self, state_id: int) -> np.ndarray:
        """Return all global frame indices assigned to ``state_id``."""

        return np.where(self.state_by_global_frame == state_id)[0]

    def to_local_indices(self, global_frame: int) -> Tuple[int, int]:
        """Map a global frame index to ``(trajectory_index, local_frame)``."""

        if global_frame < 0 or global_frame >= len(self.trajectory_index):
            raise IndexError(
                f"Global frame index {global_frame} is out of bounds for lookup of length "
                f"{len(self.trajectory_index)}."
            )
        return int(self.trajectory_index[global_frame]), int(
            self.local_frame_index[global_frame]
        )

    @property
    def n_frames(self) -> int:
        """Total number of frames represented in the lookup."""

        return int(self.state_by_global_frame.size)


def build_frame_index_lookup(dtrajs: Sequence[np.ndarray]) -> FrameIndexLookup:
    """Construct a :class:`FrameIndexLookup` from discrete trajectories."""

    if not isinstance(dtrajs, Iterable) or not dtrajs:
        raise ValueError("dtrajs must be a non-empty sequence of arrays")

    concatenated = []
    traj_indices = []
    local_indices = []
    for traj_idx, dtraj in enumerate(dtrajs):
        array = np.asarray(dtraj)
        if array.ndim != 1:
            raise ValueError("Each discrete trajectory must be one-dimensional")
        length = array.size
        concatenated.append(array)
        traj_indices.append(np.full(length, traj_idx, dtype=int))
        local_indices.append(np.arange(length, dtype=int))

    state_by_global = np.concatenate(concatenated)
    trajectory_index = np.concatenate(traj_indices)
    local_frame_index = np.concatenate(local_indices)

    return FrameIndexLookup(state_by_global, trajectory_index, local_frame_index)


RepresentativeFrame = Tuple[int, int, int, int]


class RepresentativePicker:
    """Picker for selecting representative structures from conformational states.

    Uses statistically correct weighting with TRAM/MBAR weights (not double-weighted).
    """

    def __init__(self) -> None:
        """Initialize representative picker."""

        pass

    def pick_representatives(
        self,
        features: np.ndarray,
        dtrajs: List[np.ndarray],
        state_ids: Sequence[int],
        weights: Optional[np.ndarray] = None,
        n_reps: int = 1,
        method: str = "medoid",
    ) -> List[RepresentativeFrame]:
        """Pick representative frames for given states.

        Args:
            features: Feature matrix (n_frames x n_features)
            dtrajs: Discrete trajectories
            state_ids: States to pick representatives for
            weights: Per-frame weights from TRAM/MBAR (NOT multiplied by pi)
            n_reps: Number of representatives per state
            method: Selection method ('medoid', 'centroid', 'diverse')

        Returns:
            List of ``(state_id, global_frame_index, trajectory_index, local_frame_index)``
            tuples.
        """

        lookup = build_frame_index_lookup(dtrajs)

        if features.shape[0] != lookup.n_frames:
            raise ValueError(
                "Feature matrix row count does not match total number of frames "
                f"({features.shape[0]} != {lookup.n_frames})."
            )

        if weights is not None and weights.shape[0] != lookup.n_frames:
            raise ValueError(
                "Weights vector length does not match total number of frames "
                f"({weights.shape[0]} != {lookup.n_frames})."
            )

        if method == "medoid":
            return self._pick_medoids(features, lookup, state_ids, weights, n_reps)
        if method == "centroid":
            return self._pick_centroids(features, lookup, state_ids, weights, n_reps)
        if method == "diverse":
            return self._pick_diverse(features, lookup, state_ids, weights, n_reps)

        raise ValueError(f"Unknown method: {method}")

    def _pick_medoids(
        self,
        features: np.ndarray,
        lookup: FrameIndexLookup,
        state_ids: Sequence[int],
        weights: Optional[np.ndarray],
        n_reps: int,
    ) -> List[RepresentativeFrame]:
        """Pick representatives using weighted k-medoids."""

        representatives: List[RepresentativeFrame] = []

        for state in state_ids:
            frames_in_state = lookup.frames_for_state(int(state))
            if frames_in_state.size == 0:
                raise ValueError(f"No frames found for state {state}")

            state_features = features[frames_in_state]

            if weights is not None:
                state_weights = weights[frames_in_state]
                weight_sum = float(np.sum(state_weights))
                if weight_sum <= 0.0:
                    raise ValueError(
                        f"Non-positive weight sum for state {state}: {weight_sum}"
                    )
                state_weights = state_weights / weight_sum
            else:
                state_weights = np.full(len(frames_in_state), 1.0 / len(frames_in_state))

            centroid = np.average(state_features, axis=0, weights=state_weights)

            distances = np.linalg.norm(state_features - centroid, axis=1)

            n_select = min(n_reps, len(frames_in_state))
            if n_select <= 0:
                continue

            selected_local_indices = np.argpartition(distances, n_select - 1)[
                :n_select
            ]

            for local_idx in selected_local_indices:
                global_frame = int(frames_in_state[local_idx])
                traj_idx, local_frame = lookup.to_local_indices(global_frame)
                representatives.append(
                    (int(state), global_frame, int(traj_idx), int(local_frame))
                )

        logger.info(f"Selected {len(representatives)} representatives")

        return representatives

    def _pick_centroids(
        self,
        features: np.ndarray,
        lookup: FrameIndexLookup,
        state_ids: Sequence[int],
        weights: Optional[np.ndarray],
        n_reps: int,
    ) -> List[RepresentativeFrame]:
        """Pick frames closest to weighted centroid."""

        return self._pick_medoids(features, lookup, state_ids, weights, n_reps)

    def _pick_diverse(
        self,
        features: np.ndarray,
        lookup: FrameIndexLookup,
        state_ids: Sequence[int],
        weights: Optional[np.ndarray],
        n_reps: int,
    ) -> List[RepresentativeFrame]:
        """Pick diverse representatives using max-min selection."""

        representatives: List[RepresentativeFrame] = []

        for state in state_ids:
            frames_in_state = lookup.frames_for_state(int(state))
            if frames_in_state.size == 0:
                raise ValueError(f"No frames found for state {state}")

            state_features = features[frames_in_state]
            n_select = min(n_reps, len(frames_in_state))
            if n_select <= 0:
                continue

            selected_indices: List[int] = []

            if weights is not None:
                state_weights = weights[frames_in_state]
                centroid = np.average(state_features, axis=0, weights=state_weights)
            else:
                centroid = np.mean(state_features, axis=0)

            distances_to_centroid = np.linalg.norm(state_features - centroid, axis=1)
            first_idx = int(np.argmin(distances_to_centroid))
            selected_indices.append(first_idx)

            for _ in range(n_select - 1):
                min_distances = np.full(len(state_features), np.inf)

                for sel_idx in selected_indices:
                    distances = np.linalg.norm(
                        state_features - state_features[sel_idx], axis=1
                    )
                    min_distances = np.minimum(min_distances, distances)

                next_idx = int(np.argmax(min_distances))
                selected_indices.append(next_idx)

            for local_idx in selected_indices:
                global_frame = int(frames_in_state[local_idx])
                traj_idx, local_frame = lookup.to_local_indices(global_frame)
                representatives.append(
                    (int(state), global_frame, int(traj_idx), int(local_frame))
                )

        logger.info(f"Selected {len(representatives)} diverse representatives")

        return representatives

    def pick_from_committor_range(
        self,
        committor: np.ndarray,
        features: np.ndarray,
        dtrajs: List[np.ndarray],
        committor_range: Tuple[float, float] = (0.4, 0.6),
        n_reps: int = 5,
        weights: Optional[np.ndarray] = None,
    ) -> List[RepresentativeFrame]:
        """Pick representative transition states from committor range."""

        ts_states = np.where(
            (committor >= committor_range[0]) & (committor <= committor_range[1])
        )[0]

        if len(ts_states) == 0:
            raise ValueError(
                f"No states found in committor range {committor_range}"
            )

        logger.info(
            f"Found {len(ts_states)} transition states in committor range {committor_range}"
        )

        return self.pick_representatives(
            features, dtrajs, ts_states, weights=weights, n_reps=n_reps, method="diverse"
        )

    def pick_from_flux(
        self,
        flux_matrix: np.ndarray,
        features: np.ndarray,
        dtrajs: List[np.ndarray],
        top_n: int = 10,
        n_reps_per_state: int = 1,
        weights: Optional[np.ndarray] = None,
    ) -> List[RepresentativeFrame]:
        """Pick representatives from high-flux bottleneck states."""

        flux_through_state = np.sum(flux_matrix, axis=1) + np.sum(flux_matrix, axis=0)

        bottleneck_states = np.argsort(flux_through_state)[::-1][:top_n]

        logger.info(f"Selecting from top {top_n} bottleneck states")

        return self.pick_representatives(
            features,
            dtrajs,
            bottleneck_states,
            weights=weights,
            n_reps=n_reps_per_state,
            method="medoid",
        )

    def extract_structures(
        self,
        representatives: List[RepresentativeFrame],
        trajectories: Any,
        output_dir: str,
        prefix: str = "state",
    ) -> List[str]:
        """Extract and save representative structures as PDB files."""

        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []

        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        for state_id, global_idx, traj_idx, local_idx in representatives:
            if traj_idx is None:
                raise ValueError(
                    f"Representative for state {state_id} is missing trajectory index"
                )
            if traj_idx < 0 or traj_idx >= len(trajectories):
                raise IndexError(
                    f"Trajectory index {traj_idx} is out of bounds for state {state_id}"
                )

            traj = trajectories[traj_idx]

            if local_idx is None:
                raise ValueError(
                    f"Representative for state {state_id} is missing local frame index"
                )
            if local_idx < 0 or local_idx >= len(traj):
                raise IndexError(
                    f"Local frame {local_idx} out of bounds for trajectory {traj_idx}"
                )

            frame = traj[local_idx]

            filename = f"{prefix}_{state_id:03d}_{global_idx:06d}.pdb"
            filepath = output_path / filename
            frame.save_pdb(str(filepath))

            saved_files.append(str(filepath))

        logger.info(f"Saved {len(saved_files)} structures to {output_dir}")

        return saved_files
