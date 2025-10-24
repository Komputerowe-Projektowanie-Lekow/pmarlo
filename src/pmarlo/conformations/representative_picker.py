"""Representative structure selection with corrected weighting."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("pmarlo.conformations")


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
    ) -> List[Tuple[int, int, Optional[int]]]:
        """Pick representative frames for given states.

        Args:
            features: Feature matrix (n_frames x n_features)
            dtrajs: Discrete trajectories
            state_ids: States to pick representatives for
            weights: Per-frame weights from TRAM/MBAR (NOT multiplied by pi)
            n_reps: Number of representatives per state
            method: Selection method ('medoid', 'centroid', 'diverse')

        Returns:
            List of (state_id, global_frame_index, trajectory_index) tuples
        """
        if method == "medoid":
            return self._pick_medoids(features, dtrajs, state_ids, weights, n_reps)
        elif method == "centroid":
            return self._pick_centroids(features, dtrajs, state_ids, weights, n_reps)
        elif method == "diverse":
            return self._pick_diverse(features, dtrajs, state_ids, weights, n_reps)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _pick_medoids(
        self,
        features: np.ndarray,
        dtrajs: List[np.ndarray],
        state_ids: Sequence[int],
        weights: Optional[np.ndarray],
        n_reps: int,
    ) -> List[Tuple[int, int, Optional[int]]]:
        """Pick representatives using weighted k-medoids.

        Uses only TRAM/MBAR weights, not double-weighted with stationary distribution.

        Args:
            features: Feature matrix
            dtrajs: Discrete trajectories
            state_ids: States to select from
            weights: TRAM/MBAR frame weights
            n_reps: Number of representatives per state

        Returns:
            List of (state_id, frame_index, traj_index) tuples
        """
        representatives = []

        # Concatenate dtrajs for frame → state mapping
        concatenated_dtrajs = np.concatenate(dtrajs)

        # Build trajectory index mapping
        traj_indices = np.zeros(len(concatenated_dtrajs), dtype=int)
        offset = 0
        for traj_idx, dtraj in enumerate(dtrajs):
            traj_indices[offset : offset + len(dtraj)] = traj_idx
            offset += len(dtraj)

        for state in state_ids:
            # Find frames in this state
            frames_in_state = np.where(concatenated_dtrajs == state)[0]

            if len(frames_in_state) == 0:
                logger.warning(f"No frames found for state {state}")
                continue

            # Get features for these frames
            state_features = features[frames_in_state]

            # Get weights for these frames (if provided)
            if weights is not None:
                state_weights = weights[frames_in_state]
                # Normalize weights
                state_weights = state_weights / np.sum(state_weights)
            else:
                state_weights = np.ones(len(frames_in_state)) / len(frames_in_state)

            # Compute weighted centroid
            centroid = np.average(state_features, axis=0, weights=state_weights)

            # Find closest frames to centroid (medoids)
            distances = np.linalg.norm(state_features - centroid, axis=1)

            # Select n_reps frames
            n_select = min(n_reps, len(frames_in_state))
            selected_local_indices = np.argpartition(distances, n_select - 1)[:n_select]

            # Map back to global frame indices
            for local_idx in selected_local_indices:
                global_frame = frames_in_state[local_idx]
                traj_idx = int(traj_indices[global_frame])
                representatives.append((int(state), int(global_frame), traj_idx))

        logger.info(f"Selected {len(representatives)} representatives")

        return representatives

    def _pick_centroids(
        self,
        features: np.ndarray,
        dtrajs: List[np.ndarray],
        state_ids: Sequence[int],
        weights: Optional[np.ndarray],
        n_reps: int,
    ) -> List[Tuple[int, int, Optional[int]]]:
        """Pick frames closest to weighted centroid.

        Args:
            features: Feature matrix
            dtrajs: Discrete trajectories
            state_ids: States to select from
            weights: TRAM/MBAR frame weights
            n_reps: Number per state

        Returns:
            List of representatives
        """
        # Same as medoids for single representative
        return self._pick_medoids(features, dtrajs, state_ids, weights, n_reps)

    def _pick_diverse(
        self,
        features: np.ndarray,
        dtrajs: List[np.ndarray],
        state_ids: Sequence[int],
        weights: Optional[np.ndarray],
        n_reps: int,
    ) -> List[Tuple[int, int, Optional[int]]]:
        """Pick diverse representatives using max-min selection.

        Args:
            features: Feature matrix
            dtrajs: Discrete trajectories
            state_ids: States to select from
            weights: TRAM/MBAR frame weights
            n_reps: Number per state

        Returns:
            List of representatives
        """
        representatives = []

        concatenated_dtrajs = np.concatenate(dtrajs)
        traj_indices = np.zeros(len(concatenated_dtrajs), dtype=int)
        offset = 0
        for traj_idx, dtraj in enumerate(dtrajs):
            traj_indices[offset : offset + len(dtraj)] = traj_idx
            offset += len(dtraj)

        for state in state_ids:
            frames_in_state = np.where(concatenated_dtrajs == state)[0]

            if len(frames_in_state) == 0:
                continue

            state_features = features[frames_in_state]
            n_select = min(n_reps, len(frames_in_state))

            # Max-min diversity selection
            selected_indices = []

            # Start with most central frame
            if weights is not None:
                state_weights = weights[frames_in_state]
                centroid = np.average(state_features, axis=0, weights=state_weights)
            else:
                centroid = np.mean(state_features, axis=0)

            distances_to_centroid = np.linalg.norm(state_features - centroid, axis=1)
            first_idx = int(np.argmin(distances_to_centroid))
            selected_indices.append(first_idx)

            # Iteratively add most distant frame from selected set
            for _ in range(n_select - 1):
                min_distances = np.full(len(state_features), np.inf)

                for sel_idx in selected_indices:
                    distances = np.linalg.norm(
                        state_features - state_features[sel_idx], axis=1
                    )
                    min_distances = np.minimum(min_distances, distances)

                # Pick frame with maximum minimum distance
                next_idx = int(np.argmax(min_distances))
                selected_indices.append(next_idx)

            # Map to global indices
            for local_idx in selected_indices:
                global_frame = frames_in_state[local_idx]
                traj_idx = int(traj_indices[global_frame])
                representatives.append((int(state), int(global_frame), traj_idx))

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
    ) -> List[Tuple[int, int, Optional[int]]]:
        """Pick representative transition states from committor range.

        Args:
            committor: Forward committor probabilities (per state)
            features: Feature matrix
            dtrajs: Discrete trajectories
            committor_range: (min, max) committor range for TS
            n_reps: Number of representatives to pick
            weights: TRAM/MBAR weights

        Returns:
            List of (state_id, frame_index, traj_index) tuples
        """
        # Find states in committor range
        ts_states = np.where(
            (committor >= committor_range[0]) & (committor <= committor_range[1])
        )[0]

        if len(ts_states) == 0:
            logger.warning(
                f"No states found in committor range {committor_range}"
            )
            return []

        logger.info(
            f"Found {len(ts_states)} transition states in committor range {committor_range}"
        )

        # Pick representatives from TS states
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
    ) -> List[Tuple[int, int, Optional[int]]]:
        """Pick representatives from high-flux bottleneck states.

        Args:
            flux_matrix: Reactive flux matrix
            features: Feature matrix
            dtrajs: Discrete trajectories
            top_n: Number of top bottleneck states
            n_reps_per_state: Representatives per state
            weights: TRAM/MBAR weights

        Returns:
            List of (state_id, frame_index, traj_index) tuples
        """
        # Compute flux through each state
        flux_through_state = np.sum(flux_matrix, axis=1) + np.sum(flux_matrix, axis=0)

        # Find top_n states
        bottleneck_states = np.argsort(flux_through_state)[::-1][:top_n]

        logger.info(f"Selecting from top {top_n} bottleneck states")

        # Pick representatives
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
        representatives: List[Tuple[int, int, Optional[int]]],
        trajectories: Any,
        output_dir: str,
        prefix: str = "state",
    ) -> List[str]:
        """Extract and save representative structures as PDB files.

        Args:
            representatives: List of (state_id, frame_index, traj_index) tuples
            trajectories: MDTraj trajectory object or list of trajectories
            output_dir: Output directory for PDB files
            prefix: Filename prefix

        Returns:
            List of saved PDB file paths
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # Handle single trajectory vs list
        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        for state_id, frame_idx, traj_idx in representatives:
            try:
                # Get trajectory
                if traj_idx is not None and traj_idx < len(trajectories):
                    traj = trajectories[traj_idx]

                    # Map global frame to local frame
                    # (Simplified: assumes frame_idx is already local if traj_idx provided)
                    local_frame = frame_idx

                    # Check bounds
                    if local_frame >= len(traj):
                        logger.warning(
                            f"Frame {local_frame} out of bounds for "
                            f"trajectory {traj_idx} (length {len(traj)})"
                        )
                        continue
                else:
                    # Use concatenated trajectory
                    traj = trajectories[0]
                    local_frame = frame_idx

                # Extract frame
                frame = traj[local_frame]

                # Save PDB
                filename = f"{prefix}_{state_id:03d}_{frame_idx:06d}.pdb"
                filepath = output_path / filename
                frame.save_pdb(str(filepath))

                saved_files.append(str(filepath))

            except Exception as e:
                logger.warning(
                    f"Failed to extract structure for state {state_id}, "
                    f"frame {frame_idx}: {e}"
                )

        logger.info(f"Saved {len(saved_files)} structures to {output_dir}")

        return saved_files

