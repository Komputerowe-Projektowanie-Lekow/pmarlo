"""Behavioral tests for RepresentativePicker centroid and medoid selection."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.representative_picker import RepresentativePicker


class DummyLookup:
    """Minimal FrameIndexLookup stand-in for precise test control."""

    def __init__(self, state_to_frames: dict[int, list[int] | np.ndarray]) -> None:
        self._state_to_frames = {
            int(state): np.asarray(frames, dtype=int)
            for state, frames in state_to_frames.items()
        }
        # Track each global frame -> (traj_idx, local_idx). Single trajectory so idx=0.
        self._frame_to_traj: dict[int, tuple[int, int]] = {}
        for frames in self._state_to_frames.values():
            for global_idx in frames:
                global_idx = int(global_idx)
                self._frame_to_traj[global_idx] = (0, global_idx)

    def frames_for_state(self, state: int) -> np.ndarray:
        return self._state_to_frames.get(int(state), np.array([], dtype=int))

    def to_local_indices(self, global_frame: int) -> tuple[int, int]:
        return self._frame_to_traj[int(global_frame)]


@pytest.fixture()
def picker() -> RepresentativePicker:
    """Create a RepresentativePicker without running __init__."""

    return object.__new__(RepresentativePicker)


# ---------- centroid-based selection tests ----------


def test_pick_centroids_picks_sample_closest_to_mean_single_state(picker: RepresentativePicker) -> None:
    """One state whose arithmetic mean lands on an exact frame."""

    features = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])  # mean = 2 (index 2)
    lookup = DummyLookup({0: [0, 1, 2, 3, 4]})

    reps = picker._pick_centroids(  # type: ignore[protected-access]
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=None,
        n_reps=1,
    )

    assert reps == [(0, 2, 0, 2)]


def test_pick_centroids_respects_state_partition(picker: RepresentativePicker) -> None:
    """Each state should get an independent centroid selection."""

    features = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    lookup = DummyLookup({0: [0, 1, 2], 1: [3, 4, 5]})

    reps = picker._pick_centroids(  # type: ignore[protected-access]
        features=features,
        lookup=lookup,
        state_ids=[0, 1],
        weights=None,
        n_reps=1,
    )

    assert len(reps) == 2
    chosen = {state: frame for state, frame, *_ in reps}
    assert chosen[0] == 1
    assert chosen[1] == 4


def test_pick_centroids_uses_weights_for_centroid_location(picker: RepresentativePicker) -> None:
    """Weights bias the centroid toward heavier frames."""

    features = np.array([[0.0], [10.0], [100.0]])
    lookup = DummyLookup({0: [0, 1, 2]})

    reps_uniform = picker._pick_centroids(  # type: ignore[protected-access]
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=np.array([1.0, 1.0, 1.0]),
        n_reps=1,
    )
    assert reps_uniform == [(0, 1, 0, 1)]

    reps_weighted = picker._pick_centroids(  # type: ignore[protected-access]
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=np.array([1.0, 1.0, 10.0]),
        n_reps=1,
    )
    assert reps_weighted == [(0, 2, 0, 2)]


def test_pick_centroids_respects_n_reps_cap(picker: RepresentativePicker) -> None:
    """n_reps larger than available frames should not duplicate selections."""

    features = np.array([[0.0], [1.0]])
    lookup = DummyLookup({0: [0, 1]})

    reps = picker._pick_centroids(  # type: ignore[protected-access]
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=None,
        n_reps=5,
    )

    assert len(reps) == 2
    assert {frame for _, frame, *_ in reps} == {0, 1}


# ---------- medoid-based selection tests ----------


def test_pick_medoids_one_rep_is_weighted_1_median(picker: RepresentativePicker) -> None:
    """With k=1 the medoid is the weighted one-median."""

    features = np.array([[0.0], [10.0], [20.0]])  # middle point should win
    lookup = DummyLookup({0: [0, 1, 2]})

    reps = picker._pick_medoids(  # type: ignore[protected-access]
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=np.array([1.0, 1.0, 1.0]),
        n_reps=1,
    )

    assert reps == [(0, 1, 0, 1)]


def test_pick_medoids_two_clusters_give_one_medoid_per_cluster(picker: RepresentativePicker) -> None:
    """Separated clusters should each host one medoid when k=2."""

    features = np.array([[0.0], [1.0], [2.0], [100.0], [101.0], [102.0]])
    lookup = DummyLookup({0: [0, 1, 2, 3, 4, 5]})

    reps = picker._pick_medoids(  # type: ignore[protected-access]
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=np.ones(6),
        n_reps=2,
    )

    chosen = {frame for _, frame, *_ in reps}
    assert chosen & {0, 1, 2}
    assert chosen & {3, 4, 5}
    assert len(chosen) == 2


def test_pick_medoids_respects_state_partition_and_n_reps(picker: RepresentativePicker) -> None:
    """Each state caps selections at n_reps independently."""

    features = np.array([[0.0], [5.0], [10.0], [100.0], [110.0]])
    lookup = DummyLookup({0: [0, 1, 2], 1: [3, 4]})

    reps = picker._pick_medoids(  # type: ignore[protected-access]
        features=features,
        lookup=lookup,
        state_ids=[0, 1],
        weights=np.ones(5),
        n_reps=2,
    )

    states = [state for state, *_ in reps]
    assert set(states) == {0, 1}
    assert states.count(0) <= 2
    assert states.count(1) <= 2
    assert len(reps) <= 4


def test_pick_medoids_raises_for_empty_state(picker: RepresentativePicker) -> None:
    """Requesting a state with no frames should raise."""

    features = np.array([[0.0], [1.0]])
    lookup = DummyLookup({0: [0, 1]})

    with pytest.raises(ValueError):
        picker._pick_medoids(  # type: ignore[protected-access]
            features=features,
            lookup=lookup,
            state_ids=[0, 1],
            weights=np.ones(2),
            n_reps=1,
        )


def test_pick_medoids_raises_for_non_positive_weight_sum(picker: RepresentativePicker) -> None:
    """Zero-sum weights represent invalid input."""

    features = np.array([[0.0], [1.0], [2.0]])
    lookup = DummyLookup({0: [0, 1, 2]})

    with pytest.raises(ValueError):
        picker._pick_medoids(  # type: ignore[protected-access]
            features=features,
            lookup=lookup,
            state_ids=[0],
            weights=np.array([0.0, 0.0, 0.0]),
            n_reps=1,
        )
