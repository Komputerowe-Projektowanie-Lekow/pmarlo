from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmarlo.representative_picker import (
    RepresentativePicker,
    TrajectoryFrameLocator,
    TrajectorySegment,
    build_frame_index_lookup,
)


def _state_from_global(dtrajs: list[np.ndarray], global_index: int) -> int:
    """Recover MSM state from global frame index using the dtrajs layout."""

    offset = 0
    for traj in dtrajs:
        next_offset = offset + len(traj)
        if global_index < next_offset:
            return int(traj[global_index - offset])
        offset = next_offset
    raise AssertionError("Global frame index out of range for provided dtrajs")


def _state_from_traj_local(
    dtrajs: list[np.ndarray], traj_index: int, local_index: int
) -> int:
    """Return the state stored in dtrajs[traj_index][local_index]."""

    return int(dtrajs[traj_index][local_index])


@pytest.fixture
def picker() -> RepresentativePicker:
    """Return a fresh picker instance for each test."""

    return RepresentativePicker()


@pytest.mark.parametrize("method", ["medoid", "centroid", "diverse"])
def test_single_state_single_frame_trivial_case(
    picker: RepresentativePicker, method: str
) -> None:
    """
    If there is exactly one frame in a state, that frame must be picked as the
    representative, independent of method or features.
    """

    dtrajs = [np.array([0, 1, 0], dtype=int)]
    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [5.0, 5.0],
        ]
    )

    reps = picker.pick_representatives(
        features=features,
        dtrajs=dtrajs,
        state_ids=[1],
        n_reps=1,
        method=method,
    )

    assert len(reps) == 1

    state_id, global_index, traj_index, local_index = reps[0]

    assert state_id == 1
    assert _state_from_global(dtrajs, global_index) == state_id
    assert _state_from_traj_local(dtrajs, traj_index, local_index) == state_id
    assert traj_index == 0
    assert global_index == local_index == 1


def test_multiple_states_and_n_reps_respected(picker: RepresentativePicker) -> None:
    """
    For each requested state, at most n_reps distinct frames belonging to that
    state should be returned.
    """

    dtrajs = [np.array([0, 1, 0, 1], dtype=int)]
    features = np.arange(4.0).reshape(-1, 1)

    state_ids = [0, 1]
    n_reps = 2

    reps = picker.pick_representatives(
        features=features,
        dtrajs=dtrajs,
        state_ids=state_ids,
        n_reps=n_reps,
        method="medoid",
    )

    assert 1 <= len(reps) <= len(features)

    by_state: dict[int, list[int]] = {s: [] for s in state_ids}
    for state_id, global_index, traj_index, local_index in reps:
        assert state_id in state_ids
        assert _state_from_global(dtrajs, global_index) == state_id
        assert _state_from_traj_local(dtrajs, traj_index, local_index) == state_id
        by_state[state_id].append(global_index)

    for s in state_ids:
        assert len(by_state[s]) <= n_reps
        assert len(set(by_state[s])) == len(by_state[s])


def test_ignores_states_not_requested(picker: RepresentativePicker) -> None:
    """Only requested states should be represented in the output."""

    dtrajs = [np.array([0, 1, 2, 1, 0], dtype=int)]
    features = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 1.0],
            [1.5, 1.0, 0.0],
            [2.0, 2.0, 2.0],
        ]
    )

    reps = picker.pick_representatives(
        features=features,
        dtrajs=dtrajs,
        state_ids=[1, 2],
        n_reps=1,
        method="medoid",
    )

    for state_id, global_index, traj_index, local_index in reps:
        assert state_id in {1, 2}
        assert _state_from_global(dtrajs, global_index) == state_id
        assert _state_from_traj_local(dtrajs, traj_index, local_index) == state_id


def test_features_row_count_mismatch_raises(picker: RepresentativePicker) -> None:
    """Number of feature rows must match total frames in dtrajs."""

    dtrajs = [np.array([0, 1, 0], dtype=int)]
    features = np.ones((2, 4))

    with pytest.raises(ValueError):
        picker.pick_representatives(
            features=features,
            dtrajs=dtrajs,
            state_ids=[0],
            n_reps=1,
            method="medoid",
        )


def test_weights_length_mismatch_raises(picker: RepresentativePicker) -> None:
    """Length of weights must match total frames in dtrajs."""

    dtrajs = [np.array([0, 1, 0], dtype=int)]
    features = np.ones((3, 2))
    weights = np.ones(2)

    with pytest.raises(ValueError):
        picker.pick_representatives(
            features=features,
            dtrajs=dtrajs,
            state_ids=[0],
            weights=weights,
            n_reps=1,
            method="medoid",
        )


def test_unknown_method_raises(picker: RepresentativePicker) -> None:
    """Passing an unsupported method should fail fast."""

    dtrajs = [np.array([0, 1], dtype=int)]
    features = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError):
        picker.pick_representatives(
            features=features,
            dtrajs=dtrajs,
            state_ids=[0, 1],
            n_reps=1,
            method="not-a-real-method",
        )


def test_build_frame_index_lookup_maps_global_to_local() -> None:
    dtrajs = [np.array([0, 1, 0]), np.array([2, 1])]

    lookup = build_frame_index_lookup(dtrajs)

    assert np.array_equal(lookup.state_by_global_frame, np.array([0, 1, 0, 2, 1]))
    assert np.array_equal(lookup.trajectory_index, np.array([0, 0, 0, 1, 1]))
    assert np.array_equal(lookup.local_frame_index, np.array([0, 1, 2, 0, 1]))

    traj_idx, local_idx = lookup.to_local_indices(4)
    assert traj_idx == 1
    assert local_idx == 1


def test_trajectory_segment_applies_stride_multiplier() -> None:
    segment = TrajectorySegment(
        path=Path("fake.dcd"),
        start=0,
        stop=10,
        local_start=100,
        local_stride=5,
    )
    locator = TrajectoryFrameLocator(segments=(segment,))

    _, local_frame = locator.resolve(3)
    assert local_frame == 115


def test_pick_representatives_returns_local_indices() -> None:
    dtrajs = [np.array([0, 1, 0]), np.array([2, 1])]
    features = np.array([[0.0], [0.0], [1.0], [3.0], [2.0]])
    weights = np.array([1.0, 0.8, 1.0, 1.0, 0.2])

    picker = RepresentativePicker()

    representatives = picker.pick_representatives(
        features=features,
        dtrajs=dtrajs,
        state_ids=[1],
        weights=weights,
        n_reps=1,
        method="medoid",
    )

    assert representatives == [(1, 1, 0, 1)]


class _PdbFrame:
    def __init__(self, pdb_content: str) -> None:
        self._pdb_content = pdb_content

    def save_pdb(self, path: str) -> None:
        Path(path).write_text(self._pdb_content)


class _Trajectory:
    def __init__(self, frames: list[_PdbFrame]) -> None:
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, item: int) -> _PdbFrame:
        return self._frames[item]


def test_extract_structures_uses_local_indices(tmp_path: Path) -> None:
    pdb_path = Path("tests/_assets/3gd8-fixed.pdb")

    pdb_content = pdb_path.read_text()
    frames = [_PdbFrame(pdb_content) for _ in range(6)]
    trajectories = [_Trajectory(frames[:3]), _Trajectory(frames[3:6])]

    picker = RepresentativePicker()

    representatives = [(5, 4, 1, 1)]

    saved = picker.extract_structures(
        representatives,
        trajectories,
        output_dir=str(tmp_path / "structures"),
        prefix="test",
    )

    assert len(saved) == 1
    saved_path = Path(saved[0])
    assert saved_path.exists()
    assert saved_path.name == "test_005_000004.pdb"


def test_extract_structures_with_locator(tmp_path: Path) -> None:
    import mdtraj as md

    topology = Path("tests/_assets/3gd8-fixed.pdb").resolve()
    trajectory = Path("tests/_assets/traj.dcd").resolve()

    traj = md.load(str(trajectory), top=str(topology))
    locator = TrajectoryFrameLocator(
        segments=(
            TrajectorySegment(
                path=trajectory,
                start=0,
                stop=len(traj),
                local_start=0,
            ),
        )
    )

    picker = RepresentativePicker()
    representatives = [(0, 2, 0, 2)]

    saved = picker.extract_structures(
        representatives,
        trajectories=None,
        output_dir=str(tmp_path / "locator"),
        prefix="loc",
        topology_path=topology,
        trajectory_locator=locator,
    )

    assert len(saved) == 1
    saved_path = Path(saved[0])
    assert saved_path.exists()
    assert saved_path.name == "loc_000_000002.pdb"
