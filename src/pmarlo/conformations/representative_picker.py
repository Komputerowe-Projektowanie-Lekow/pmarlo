"""Representative structure selection with corrected weighting."""

"""
The file is after the analysis. All should be working with the new test suite.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("pmarlo.conformations")


@dataclass(frozen=True)
class TrajectorySegment:
    """Mapping between global frame span and a physical trajectory file."""

    path: Path
    start: int
    stop: int
    local_start: int
    local_stride: int = 1

    def __post_init__(self) -> None:
        if self.local_stride <= 0:
            raise ValueError("TrajectorySegment.local_stride must be positive")

    def contains(self, global_frame: int) -> bool:
        return self.start <= global_frame < self.stop

    def to_local(self, global_frame: int) -> int:
        offset = global_frame - self.start
        return self.local_start + offset * self.local_stride


@dataclass(frozen=True)
class TrajectoryFrameLocator:
    """Resolve global frame indices to physical trajectory files."""

    segments: Tuple[TrajectorySegment, ...]

    def resolve(self, global_frame: int) -> Tuple[Path, int]:
        for segment in self.segments:
            if segment.contains(global_frame):
                return segment.path, segment.to_local(global_frame)
        raise IndexError(
            f"Global frame {global_frame} does not map to any known trajectory segment"
        )


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


RepresentativeFrame = Tuple[int, int, Optional[int], Optional[int]]


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
            features:
                Feature matrix of shape (n_frames, n_features) covering all frames
                in all discrete trajectories.
            dtrajs:
                List of discrete trajectories whose frames are indexed in ``features``.
            state_ids:
                Iterable of MSM state ids to pick representatives for.
            weights:
                Optional 1D array of per frame weights from TRAM/MBAR
                (not pre multiplied by stationary distribution). Length must equal
                the total number of frames across all ``dtrajs``.
            n_reps:
                Number of representatives to select per state. Must be an integer
                greater or equal to 1.
            method:
                Selection method. One of:
                    - "medoid"
                    - "centroid"
                    - "diverse"

        Returns:
            List[RepresentativeFrame]:
                One representative record per selected frame. Each record encodes
                at least
                    - state id
                    - global frame index
                    - trajectory index
                    - local frame index
                according to the definition of ``RepresentativeFrame``.
        """
        # Build lookup from (traj_idx, local_frame_idx) to global frame index
        lookup = build_frame_index_lookup(dtrajs)
        n_frames = lookup.n_frames

        # Validate features
        if features.ndim != 2:
            raise ValueError(
                f"Feature matrix must be 2D (n_frames x n_features), got ndim={features.ndim}"
            )
        if features.shape[0] != n_frames:
            raise ValueError(
                "Feature matrix row count does not match total number of frames "
                f"({features.shape[0]} != {n_frames})."
            )

        # Validate weights, if provided
        if weights is not None:
            if weights.ndim != 1:
                raise ValueError(
                    f"Weights must be a 1D array of length n_frames, got ndim={weights.ndim}"
                )
            if weights.shape[0] != n_frames:
                raise ValueError(
                    "Weights vector length does not match total number of frames "
                    f"({weights.shape[0]} != {n_frames})."
                )
            if not np.all(np.isfinite(weights)):
                raise ValueError("Weights must be finite")
            if np.any(weights < 0):
                raise ValueError("Weights must be non negative")

        # Validate n_reps
        try:
            n_reps_int = int(n_reps)
        except (TypeError, ValueError) as exc:
            raise TypeError("n_reps must be an integer") from exc

        if n_reps_int < 1:
            raise ValueError(f"n_reps must be >= 1, got {n_reps_int}")
        n_reps = n_reps_int

        # Normalise method name
        method = method.lower()

        if method == "medoid":
            return self._pick_medoids(features, lookup, state_ids, weights, n_reps)
        if method == "centroid":
            return self._pick_centroids(features, lookup, state_ids, weights, n_reps)
        if method == "diverse":
            return self._pick_diverse(features, lookup, state_ids, weights, n_reps)

        valid = ("medoid", "centroid", "diverse")
        raise ValueError(f"Unknown method '{method}'. Expected one of {valid}")

    def _pick_medoids(
    self,
    features: np.ndarray,
    lookup: FrameIndexLookup,
    state_ids: Sequence[int],
    weights: Optional[np.ndarray],
    n_reps: int,
) -> List[RepresentativeFrame]:
        """Pick representatives using weighted k-medoids within each state.

        For each MSM state, we:
        - restrict to frames in that state,
        - run a weighted k-medoids clustering in feature space,
        - choose the medoid indices as representative frames.

        The objective minimized is the weighted sum of distances to the nearest medoid.
        The medoids are always actual frames from the state.
        """

        representatives: List[RepresentativeFrame] = []

        for state in state_ids:
            frames_in_state = lookup.frames_for_state(int(state))
            if frames_in_state.size == 0:
                raise ValueError(f"No frames found for state {state}")

            state_features = features[frames_in_state]
            n_frames = state_features.shape[0]

            n_select = min(n_reps, n_frames)
            if n_select <= 0:
                continue

            # Build weights for this state
            if weights is not None:
                state_weights = np.asarray(weights[frames_in_state], dtype=float)
                if np.any(state_weights < 0.0):
                    raise ValueError(
                        f"Negative weights encountered in state {state}"
                    )
                weight_sum = float(np.sum(state_weights))
                if weight_sum <= 0.0:
                    raise ValueError(
                        f"Non-positive weight sum for state {state}: {weight_sum}"
                    )
                # Normalization is not strictly required, but keeps scales reasonable
                state_weights = state_weights / weight_sum
            else:
                state_weights = np.full(n_frames, 1.0 / n_frames, dtype=float)

            # Trivial case: only one frame in this state
            if n_frames == 1:
                medoid_indices = [0]
            else:
                # Precompute pairwise distance matrix within this state
                diff = state_features[:, None, :] - state_features[None, :, :]
                D = np.linalg.norm(diff, axis=2)  # shape (n_frames, n_frames)

                if n_select == 1:
                    # Single medoid: point with minimal weighted average distance
                    weighted_dist_sums = D @ state_weights
                    medoid_indices = [int(np.argmin(weighted_dist_sums))]
                else:
                    # Run a simple PAM-like k-medoids:
                    # 1) initialize medoids
                    # 2) improve by local swaps

                    # Initial medoid: minimal weighted average distance
                    weighted_dist_sums = D @ state_weights
                    medoids = [int(np.argmin(weighted_dist_sums))]

                    # Add further medoids greedily as farthest weighted points
                    while len(medoids) < n_select:
                        dist_to_nearest = np.min(D[:, medoids], axis=1)  # (n_frames,)
                        scores = state_weights * dist_to_nearest

                        # Avoid picking an already chosen medoid
                        for m_idx in medoids:
                            scores[m_idx] = -1.0

                        next_medoid = int(np.argmax(scores))
                        medoids.append(next_medoid)

                    # Refinement by local swaps
                    max_iter = 50
                    m = n_frames
                    medoid_set = set(medoids)

                    for _ in range(max_iter):
                        # Current cost
                        dist_to_medoids = D[:, medoids]  # (m, k)
                        nearest_idx = np.argmin(dist_to_medoids, axis=1)
                        nearest_dist = dist_to_medoids[np.arange(m), nearest_idx]
                        current_cost = float(np.sum(state_weights * nearest_dist))

                        best_cost = current_cost
                        best_swap = None

                        # Try swapping each medoid with each non-medoid
                        for mi, m_idx in enumerate(medoids):
                            for candidate_idx in range(m):
                                if candidate_idx in medoid_set:
                                    continue

                                candidate_medoids = medoids.copy()
                                candidate_medoids[mi] = candidate_idx

                                dist_to_candidate = D[:, candidate_medoids]
                                nearest_cand = np.min(dist_to_candidate, axis=1)
                                cand_cost = float(
                                    np.sum(state_weights * nearest_cand)
                                )

                                if cand_cost + 1e-12 < best_cost:
                                    best_cost = cand_cost
                                    best_swap = (mi, candidate_idx)

                        if best_swap is None:
                            # No improving swap
                            break

                        mi, new_idx = best_swap
                        old_idx = medoids[mi]
                        medoids[mi] = new_idx
                        medoid_set.remove(old_idx)
                        medoid_set.add(new_idx)

                    medoid_indices = medoids

            # Convert local medoid indices back to global frame indices
            for local_idx in medoid_indices:
                global_frame = int(frames_in_state[local_idx])
                traj_idx, local_frame = lookup.to_local_indices(global_frame)
                representatives.append(
                    (int(state), global_frame, int(traj_idx), int(local_frame))
                )

        logger.info(f"Selected {len(representatives)} medoid representatives")

        return representatives


    def _pick_centroids(
        self,
        features: np.ndarray,
        lookup: FrameIndexLookup,
        state_ids: Sequence[int],
        weights: Optional[np.ndarray],
        n_reps: int,
    ) -> List[RepresentativeFrame]:
        """Pick frames closest to a weighted centroid within each state."""

        representatives: List[RepresentativeFrame] = []

        for state in state_ids:
            frames_in_state = lookup.frames_for_state(int(state))
            if frames_in_state.size == 0:
                raise ValueError(f"No frames found for state {state}")

            state_features = features[frames_in_state]

            # Build weights for this state
            if weights is not None:
                state_weights = np.asarray(weights[frames_in_state], dtype=float)
                weight_sum = float(np.sum(state_weights))
                if weight_sum <= 0.0:
                    raise ValueError(
                        f"Non-positive weight sum for state {state}: {weight_sum}"
                    )
                state_weights = state_weights / weight_sum
            else:
                state_weights = np.full(
                    len(frames_in_state), 1.0 / len(frames_in_state), dtype=float
                )

            # Weighted centroid in feature space
            centroid = np.average(state_features, axis=0, weights=state_weights)
            distances = np.linalg.norm(state_features - centroid, axis=1)

            n_select = min(n_reps, len(frames_in_state))
            if n_select <= 0:
                continue

            selected_local_indices = np.argpartition(
                distances, n_select - 1
            )[:n_select]

            for local_idx in selected_local_indices:
                global_frame = int(frames_in_state[local_idx])
                traj_idx, local_frame = lookup.to_local_indices(global_frame)
                representatives.append(
                    (int(state), global_frame, int(traj_idx), int(local_frame))
                )

        logger.info(f"Selected {len(representatives)} centroid representatives")

        return representatives

    def _pick_diverse(
        self,
        features: np.ndarray,
        lookup: FrameIndexLookup,
        state_ids: Sequence[int],
        weights: Optional[np.ndarray],
        n_reps: int,
    ) -> List[RepresentativeFrame]:
        """Pick diverse representatives using greedy max-min selection per state.

        Assumptions:
          - `features` has shape (n_frames, n_features) and is indexed by global frame index.
          - `lookup` uses the same global frame index space as `features`.
          - If `weights` is not None, it has length `n_frames`.
        """

        if n_reps <= 0:
            raise ValueError("n_reps must be a positive integer")

        if features.ndim != 2:
            raise ValueError("features must be a 2D array of shape (n_frames, n_features)")

        n_frames = features.shape[0]
        lookup_n_frames = getattr(lookup, "n_frames", n_frames)
        if lookup_n_frames != n_frames:
            raise ValueError(
                "Feature matrix row count does not match total number of frames "
                f"({n_frames} != {lookup_n_frames})."
            )

        weights_arr: Optional[np.ndarray]
        if weights is not None:
            weights_arr = np.asarray(weights, dtype=float)
            if weights_arr.shape[0] != n_frames:
                raise ValueError(
                    f"weights must have length {n_frames}, got {weights_arr.shape[0]}"
                )
        else:
            weights_arr = None

        representatives: List[RepresentativeFrame] = []

        # Preserve order but avoid processing the same state multiple times
        unique_states = list(dict.fromkeys(int(s) for s in state_ids))

        for state in unique_states:
            frames_in_state = lookup.frames_for_state(int(state))
            if frames_in_state.size == 0:
                raise ValueError(f"No frames found for state {state}")

            state_features = features[frames_in_state]
            n_available = state_features.shape[0]
            n_select = min(n_reps, n_available)
            if n_select <= 0:
                continue

            # Compute centroid of the state in feature space
            if weights_arr is not None:
                state_weights = weights_arr[frames_in_state]
                if np.all(state_weights == 0.0):
                    # Degenerate weights, fall back to unweighted mean
                    centroid = np.mean(state_features, axis=0)
                else:
                    centroid = np.average(state_features, axis=0, weights=state_weights)
            else:
                centroid = np.mean(state_features, axis=0)

            # First representative: closest to centroid (medoid-like)
            distances_to_centroid = np.linalg.norm(state_features - centroid, axis=1)
            first_idx = int(np.argmin(distances_to_centroid))
            selected_indices: List[int] = [first_idx]

            # Additional representatives: greedy max-min selection
            for _ in range(n_select - 1):
                # For each candidate frame, compute distance to closest already selected frame
                min_distances = np.full(n_available, np.inf)
                for sel_idx in selected_indices:
                    distances = np.linalg.norm(
                        state_features - state_features[sel_idx],
                        axis=1,
                    )
                    min_distances = np.minimum(min_distances, distances)

                # Prevent re-selecting already chosen indices
                if selected_indices:
                    min_distances[selected_indices] = -np.inf

                next_idx = int(np.argmax(min_distances))
                if next_idx in selected_indices or not np.isfinite(min_distances[next_idx]):
                    # No further diversity to gain (all remaining points are identical or already selected)
                    break

                selected_indices.append(next_idx)

            # Convert local indices back to global and trajectory indices
            for local_idx in selected_indices:
                global_frame = int(frames_in_state[local_idx])
                traj_idx, local_frame = lookup.to_local_indices(global_frame)
                representatives.append(
                    (int(state), global_frame, int(traj_idx), int(local_frame))
                )

        logger.info(
            "Selected %d diverse representatives from %d states",
            len(representatives),
            len(unique_states),
        )

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
        """Pick representative transition states from a committor range.

        Assumptions / contracts:
        - `committor` is a 1D array of per-state committor values in [0, 1].
        - `committor_range = (q_min, q_max)` with 0 <= q_min <= q_max <= 1.
        - State indices present in `dtrajs` are in [0, len(committor) - 1].
        - `n_reps` is a positive integer.
        """

        # Normalize and validate committor
        committor = np.asarray(committor, dtype=float)
        if committor.ndim != 1:
            raise ValueError(
                "committor must be a 1D array of per-state values; "
                f"got array with shape {committor.shape}"
            )

        # Validate committor_range
        if len(committor_range) != 2:
            raise ValueError(
                f"committor_range must be a 2-tuple (q_min, q_max), got {committor_range}"
            )
        q_min, q_max = float(committor_range[0]), float(committor_range[1])
        if not (0.0 <= q_min <= q_max <= 1.0):
            raise ValueError(
                "committor_range must satisfy 0.0 <= q_min <= q_max <= 1.0; "
                f"got ({q_min}, {q_max})"
            )

        # Basic check on n_reps
        if n_reps <= 0:
            raise ValueError(f"n_reps must be a positive integer; got {n_reps}")

        # Normalize dtrajs so attribute access (e.g., .size) is safe and check dimensionality.
        normalized_dtrajs: List[np.ndarray] = []
        for idx, dtraj in enumerate(dtrajs):
            array = np.asarray(dtraj)
            if array.ndim != 1:
                raise ValueError(
                    "Each discrete trajectory must be one-dimensional for committor selection; "
                    f"trajectory {idx} has shape {array.shape}."
                )
            normalized_dtrajs.append(array)

        # Optional sanity check that dtrajs are compatible with committor
        if normalized_dtrajs:
            non_empty = [dt for dt in normalized_dtrajs if dt.size > 0]
            if non_empty:
                min_state = min(int(np.min(dt)) for dt in non_empty)
                max_state = max(int(np.max(dt)) for dt in non_empty)
                n_states = committor.shape[0]
                if min_state < 0 or max_state >= n_states:
                    raise ValueError(
                        "dtrajs contain state indices outside the range defined by committor: "
                        f"min={min_state}, max={max_state}, allowed=[0, {n_states - 1}]"
                    )

        # Select states whose committor lies in the TS window
        ts_mask = (committor >= q_min) & (committor <= q_max)
        ts_states = np.nonzero(ts_mask)[0]

        if ts_states.size == 0:
            raise ValueError(f"No states found in committor range ({q_min}, {q_max})")

        logger.info(
            "Found %d transition states in committor range (%.3f, %.3f)",
            ts_states.size,
            q_min,
            q_max,
        )

        # Delegate to the generic representative picker for diversity-based selection
        return self.pick_representatives(
            features,
            normalized_dtrajs,
            ts_states,
            weights=weights,
            n_reps=n_reps,
            method="diverse",
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
        """Pick representatives from high-flux states based on a TPT flux matrix.

        Definition of "flux through state i":
            F_i = 0.5 * (sum_j flux[i, j] + sum_j flux[j, i])

        The algorithm:
        1. Validate the flux matrix shape.
        2. Compute F_i for each MSM state.
        3. Rank states by F_i in descending order.
        4. Select the top `top_n` states (clamped to the number of states).
        5. Use `pick_representatives` to pick `n_reps_per_state` medoids
            per selected state.
        """

        # Basic shape checks for the flux matrix
        if flux_matrix.ndim != 2:
            raise ValueError(
                f"flux_matrix must be 2D, got array with ndim={flux_matrix.ndim}"
            )

        n_states_in, n_states_out = flux_matrix.shape
        if n_states_in != n_states_out:
            raise ValueError(
                "flux_matrix must be square; "
                f"got shape {flux_matrix.shape}"
            )

        n_states = n_states_in

        if top_n is None:
            raise ValueError("top_n must be an integer >= 1, got None")

        if top_n <= 0:
            raise ValueError(f"top_n must be >= 1, got {top_n}")

        if n_reps_per_state <= 0:
            raise ValueError(
                f"n_reps_per_state must be >= 1, got {n_reps_per_state}"
            )

        # Compute symmetric "flux through state"
        flux_through_state = 0.5 * (
            np.sum(flux_matrix, axis=1) + np.sum(flux_matrix, axis=0)
        )

        # Clamp top_n to the number of states if needed
        actual_top_n = min(top_n, n_states)
        if actual_top_n < top_n:
            logger.info(
                "Requested top_n=%d bottleneck states, but only %d states are "
                "available. Using top_n=%d.",
                top_n,
                n_states,
                actual_top_n,
            )

        # Rank states by flux, descending, and take the top ones
        bottleneck_states = np.argsort(flux_through_state)[::-1][:actual_top_n]

        logger.info(
            "Selecting representatives from top %d high-flux states",
            actual_top_n,
        )

        # Delegate the actual frame selection to the generic picker
        return self.pick_representatives(
            features=features,
            dtrajs=dtrajs,
            states=bottleneck_states,
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
        *,
        topology_path: str | Path | None = None,
        trajectory_locator: TrajectoryFrameLocator | None = None,
    ) -> List[str]:
        """Extract and save representative structures as PDB files.

        Two modes:
        - Raw-trajectory mode: if ``trajectory_locator`` is provided, trajectories
            are resolved from disk using global frame indices and ``mdtraj``.
        - In-memory mode: if ``trajectory_locator`` is None, ``trajectories``
            must be provided and support indexing + ``save_pdb`` on frames.
        """

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []

        # -------------------------
        # Mode 1: raw trajectories
        # -------------------------
        if trajectory_locator is not None:
            if topology_path is None:
                raise ValueError(
                    "topology_path is required when extracting structures from raw trajectories"
                )

            top_path = Path(topology_path).resolve()
            if not top_path.exists():
                raise FileNotFoundError(
                    f"Topology file {top_path} required for representative extraction does not exist"
                )

            try:
                import mdtraj as md  # type: ignore[import]
            except ImportError as exc:  # pragma: no cover - optional dependency guard
                raise ImportError(
                    "mdtraj is required to extract structures from raw trajectories. "
                    "Install it with `pip install mdtraj`."
                ) from exc

            for state_id, global_idx, _traj_idx, _local_idx in representatives:
                traj_path, frame_index = trajectory_locator.resolve(global_idx)

                if not traj_path.exists():
                    raise FileNotFoundError(
                        f"Trajectory file {traj_path} does not exist for state {state_id}"
                    )

                frame = md.load_frame(
                    str(traj_path), index=int(frame_index), top=str(top_path)
                )

                filename = f"{prefix}_{state_id:03d}_{global_idx:06d}.pdb"
                filepath = output_path / filename
                frame.save_pdb(str(filepath))
                saved_files.append(str(filepath))

            logger.info("Saved %d structures to %s", len(saved_files), output_dir)
            return saved_files

        # -------------------------
        # Mode 2: in-memory trajectories
        # -------------------------
        if trajectories is None:
            raise ValueError(
                "trajectories must be provided when trajectory_locator is None"
            )

        # Normalize to a list
        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        if len(trajectories) == 0:
            raise ValueError("No trajectories provided for representative extraction")

        for state_id, global_idx, traj_idx, local_idx in representatives:
            if traj_idx is None:
                raise ValueError(
                    f"Representative for state {state_id} is missing trajectory index"
                )
            if not (0 <= traj_idx < len(trajectories)):
                raise IndexError(
                    f"Trajectory index {traj_idx} is out of bounds for state {state_id}"
                )

            traj = trajectories[traj_idx]

            if local_idx is None:
                raise ValueError(
                    f"Representative for state {state_id} is missing local frame index"
                )
            if not (0 <= local_idx < len(traj)):
                raise IndexError(
                    f"Local frame {local_idx} out of bounds for trajectory {traj_idx}"
                )

            frame = traj[local_idx]

            # Assumes `frame` exposes `.save_pdb(path)` (e.g. mdtraj.Trajectory slice)
            filename = f"{prefix}_{state_id:03d}_{global_idx:06d}.pdb"
            filepath = output_path / filename
            frame.save_pdb(str(filepath))

            saved_files.append(str(filepath))

        logger.info("Saved %d structures to %s", len(saved_files), output_dir)
        return saved_files
