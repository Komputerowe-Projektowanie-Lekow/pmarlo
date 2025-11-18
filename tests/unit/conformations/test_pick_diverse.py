from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.representative_picker import RepresentativePicker


class DummyLookup:
    """Minimal lookup object to exercise `_pick_diverse` without real trajectories."""

    def __init__(self, frame_states, traj_lengths=None):
        self.frame_states = np.asarray(frame_states, dtype=int)
        self.n_frames = self.frame_states.shape[0]

        if traj_lengths is None:
            self._traj_lengths = [self.n_frames]
            self._traj_starts = [0]
        else:
            self._traj_lengths = list(traj_lengths)
            self._traj_starts = []
            start = 0
            for length in self._traj_lengths:
                self._traj_starts.append(start)
                start += length
            if start != self.n_frames:
                raise ValueError("traj_lengths must sum to the number of frames")

    def frames_for_state(self, state):
        return np.where(self.frame_states == int(state))[0]

    def to_local_indices(self, global_frame):
        g = int(global_frame)
        for traj_idx, (start, length) in enumerate(
            zip(self._traj_starts, self._traj_lengths)
        ):
            if start <= g < start + length:
                return traj_idx, g - start
        raise ValueError(f"Global frame {global_frame} out of range")


def _call_pick_diverse(features, lookup, state_ids, weights, n_reps):
    """Invoke the private implementation without depending on `__init__`."""

    dummy_self = object()
    return RepresentativePicker._pick_diverse(
        dummy_self,
        features=features,
        lookup=lookup,
        state_ids=state_ids,
        weights=weights,
        n_reps=n_reps,
    )


def test_pick_diverse_single_state_diversity_unweighted():
    features = np.array([[0.0], [1.0], [10.0]], dtype=float)
    frame_states = [0, 0, 0]
    lookup = DummyLookup(frame_states)

    reps = _call_pick_diverse(
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=None,
        n_reps=2,
    )

    assert len(reps) == 2
    states = {r[0] for r in reps}
    assert states == {0}

    global_frames = sorted(r[1] for r in reps)
    assert global_frames == [0, 2]


def test_pick_diverse_respects_n_reps_and_state_membership():
    frame_states = [0, 0, 1, 1, 1]
    features = np.arange(5, dtype=float).reshape(-1, 1)
    lookup = DummyLookup(frame_states)

    reps = _call_pick_diverse(
        features=features,
        lookup=lookup,
        state_ids=[0, 1],
        weights=None,
        n_reps=2,
    )

    by_state = {}
    for state_id, global_frame, traj_idx, local_idx in reps:
        by_state.setdefault(state_id, []).append(global_frame)
        assert frame_states[global_frame] == state_id
        assert traj_idx == 0  # DummyLookup exposes a single trajectory
        assert local_idx == global_frame

    assert len(by_state[0]) == 2
    assert len(by_state[1]) == 2
    assert set(by_state) == {0, 1}


def test_pick_diverse_weighted_centroid_shifts_first_choice():
    features = np.array([[0.0], [10.0], [11.0]], dtype=float)
    frame_states = [0, 0, 0]
    lookup = DummyLookup(frame_states)

    reps_unweighted = _call_pick_diverse(
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=None,
        n_reps=2,
    )
    assert reps_unweighted[0][1] == 1

    weights = np.array([0.1, 0.1, 10.0], dtype=float)
    reps_weighted = _call_pick_diverse(
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=weights,
        n_reps=2,
    )
    assert reps_weighted[0][1] == 2


def test_pick_diverse_multi_state_uses_lookup_to_map_traj_and_local_indices():
    frame_states = [0, 0, 0, 0, 0]
    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.4, 0.0],
        ],
        dtype=float,
    )
    lookup = DummyLookup(frame_states, traj_lengths=[2, 3])

    reps = _call_pick_diverse(
        features=features,
        lookup=lookup,
        state_ids=[0],
        weights=None,
        n_reps=3,
    )

    assert len(reps) == 3
    for _state_id, global_frame, traj_idx, local_idx in reps:
        if traj_idx == 0:
            expected_global = local_idx
        elif traj_idx == 1:
            expected_global = 2 + local_idx
        else:
            pytest.fail("Unexpected trajectory index")
        assert expected_global == global_frame


def test_pick_diverse_raises_when_state_has_no_frames():
    frame_states = [0, 0]
    features = np.zeros((2, 1), dtype=float)
    lookup = DummyLookup(frame_states)

    with pytest.raises(ValueError, match="No frames found for state 1"):
        _call_pick_diverse(
            features=features,
            lookup=lookup,
            state_ids=[0, 1],
            weights=None,
            n_reps=1,
        )


def test_pick_diverse_rejects_nonpositive_n_reps():
    frame_states = [0, 0]
    features = np.zeros((2, 1), dtype=float)
    lookup = DummyLookup(frame_states)

    with pytest.raises(ValueError, match="n_reps must be a positive integer"):
        _call_pick_diverse(
            features=features,
            lookup=lookup,
            state_ids=[0],
            weights=None,
            n_reps=0,
        )
