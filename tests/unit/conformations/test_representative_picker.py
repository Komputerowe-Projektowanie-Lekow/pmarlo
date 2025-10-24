from __future__ import annotations

from pathlib import Path

import numpy as np

from pmarlo.conformations.representative_picker import (
    RepresentativePicker,
    build_frame_index_lookup,
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
