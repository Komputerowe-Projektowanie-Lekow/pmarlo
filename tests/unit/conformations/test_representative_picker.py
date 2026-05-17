from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pmarlo.conformations.representative_picker import (
    RepresentativePicker,
    TrajectoryFrameLocator,
    TrajectorySegment,
    build_frame_index_lookup,
)


def _load_test_trajectory() -> md.Trajectory:
    topology = Path("tests/_assets/3gd8-fixed.pdb").resolve()
    trajectory = Path("tests/_assets/traj.dcd").resolve()
    return md.load(str(trajectory), top=str(topology))


def _build_distance_features(traj: md.Trajectory) -> np.ndarray:
    distances = np.linalg.norm(traj.xyz[:, 0, :] - traj.xyz[:, 1, :], axis=1)
    return distances.reshape(-1, 1)


def _build_state_assignments(features: np.ndarray) -> np.ndarray:
    threshold = float(np.median(features[:, 0]))
    return np.where(features[:, 0] <= threshold, 0, 1).astype(int)


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
        method="closest_to_centroid",
    )

    assert representatives == [(1, 1, 0, 1)]


def test_pick_true_medoid_minimizes_total_distance() -> None:
    traj = _load_test_trajectory()
    features = _build_distance_features(traj)
    dtrajs = [_build_state_assignments(features)]

    picker = RepresentativePicker()
    representatives = picker.pick_representatives(
        features=features,
        dtrajs=dtrajs,
        state_ids=[0],
        n_reps=1,
        method="true_medoid",
    )

    frames_in_state = np.where(dtrajs[0] == 0)[0]
    state_features = features[frames_in_state]
    distance_sums = np.array(
        [
            np.sum(np.linalg.norm(state_features - state_features[idx], axis=1))
            for idx in range(len(state_features))
        ]
    )
    expected_local = int(np.argmin(distance_sums))
    expected_global = int(frames_in_state[expected_local])

    assert representatives == [(0, expected_global, 0, expected_global)]


def test_diverse_selection_avoids_duplicate_frames() -> None:
    traj = _load_test_trajectory()
    n_frames = traj.n_frames
    assert n_frames >= 2

    constant_value = float(np.mean(traj.xyz))
    features = np.full((n_frames, 1), constant_value)
    dtrajs = [np.zeros(n_frames, dtype=int)]

    picker = RepresentativePicker()
    n_reps = min(3, n_frames)
    representatives = picker.pick_representatives(
        features=features,
        dtrajs=dtrajs,
        state_ids=[0],
        n_reps=n_reps,
        method="diverse",
    )

    global_frames = [rep[1] for rep in representatives]
    assert len(global_frames) == len(set(global_frames))


def test_diverse_rejects_non_finite_weights() -> None:
    traj = _load_test_trajectory()
    features = _build_distance_features(traj)
    dtrajs = [_build_state_assignments(features)]
    weights = features[:, 0].copy()

    frames_in_state = np.where(dtrajs[0] == 0)[0]
    assert frames_in_state.size > 0
    weights[frames_in_state[0]] = np.nan

    picker = RepresentativePicker()

    with pytest.raises(ValueError, match="Non-finite weights"):
        picker.pick_representatives(
            features=features,
            dtrajs=dtrajs,
            state_ids=[0],
            weights=weights,
            n_reps=1,
            method="diverse",
        )


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
