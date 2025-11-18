"""Functional tests for :meth:`RepresentativePicker.extract_structures`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import sys

import pytest

from pmarlo.conformations.representative_picker import RepresentativePicker

# RepresentativeFrame = Tuple[state_id, global_idx, traj_idx, local_idx]
RepresentativeFrame = Tuple[int, int, int | None, int | None]


class FakeFrame:
    """Minimal frame that records saved files to mimic mdtraj frames."""

    def __init__(self) -> None:
        self.saved_paths: List[str] = []

    def save_pdb(self, path: str) -> None:
        output = Path(path)
        output.write_text("ATOM 1 C DUM 1 0.0 0.0 0.0\n")
        self.saved_paths.append(path)


class FakeTrajectory:
    """Simple trajectory supporting ``len`` and ``__getitem__``."""

    def __init__(self, n_frames: int) -> None:
        self._frames = [FakeFrame() for _ in range(n_frames)]

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> FakeFrame:
        return self._frames[idx]


class FakeTrajectoryLocator:
    """Maps global frame indices to a single trajectory file."""

    def __init__(self, traj_path: Path, fixed_index: int = 0) -> None:
        self.traj_path = traj_path
        self.fixed_index = fixed_index
        self.calls: List[int] = []

    def resolve(self, global_idx: int) -> Tuple[Path, int]:
        self.calls.append(global_idx)
        return self.traj_path, self.fixed_index


@pytest.fixture
def extractor() -> RepresentativePicker:
    """Provide a fresh picker per test."""

    return RepresentativePicker()


def test_in_memory_basic_two_representatives(tmp_path: Path, extractor: RepresentativePicker) -> None:
    """Ensure in-memory extraction writes deterministic names."""

    traj = FakeTrajectory(n_frames=5)
    reps: List[RepresentativeFrame] = [
        (0, 10, 0, 1),
        (1, 20, 0, 3),
    ]

    out_dir = tmp_path / "pdbs"
    result = extractor.extract_structures(
        representatives=reps,
        trajectories=[traj],
        output_dir=str(out_dir),
        prefix="state",
        topology_path=None,
        trajectory_locator=None,
    )

    assert len(result) == 2
    for path, (state_id, global_idx, _, _) in zip(result, reps):
        saved = Path(path)
        assert saved.exists()
        assert saved.parent == out_dir
        assert saved.name == f"state_{state_id:03d}_{global_idx:06d}.pdb"
        assert "ATOM 1 C DUM" in saved.read_text()


def test_in_memory_accepts_single_trajectory(tmp_path: Path, extractor: RepresentativePicker) -> None:
    """Passing a trajectory object directly still works."""

    traj = FakeTrajectory(n_frames=3)
    reps: List[RepresentativeFrame] = [
        (0, 5, 0, 2),
    ]

    out_dir = tmp_path / "single_traj"
    result = extractor.extract_structures(
        representatives=reps,
        trajectories=traj,
        output_dir=str(out_dir),
        prefix="rep",
        topology_path=None,
        trajectory_locator=None,
    )

    assert len(result) == 1
    saved = Path(result[0])
    assert saved.exists()
    assert saved.name.startswith("rep_")
    assert saved.suffix == ".pdb"


def test_in_memory_raises_on_out_of_bounds_traj_index(tmp_path: Path, extractor: RepresentativePicker) -> None:
    """Invalid trajectory index should raise an IndexError."""

    traj = FakeTrajectory(n_frames=3)
    reps: List[RepresentativeFrame] = [
        (0, 10, 1, 0),
    ]

    with pytest.raises(IndexError):
        extractor.extract_structures(
            representatives=reps,
            trajectories=[traj],
            output_dir=str(tmp_path / "out"),
            prefix="state",
            topology_path=None,
            trajectory_locator=None,
        )


def test_in_memory_raises_on_out_of_bounds_local_index(tmp_path: Path, extractor: RepresentativePicker) -> None:
    """Invalid local frame index should raise an IndexError."""

    traj = FakeTrajectory(n_frames=2)
    reps: List[RepresentativeFrame] = [
        (0, 10, 0, 5),
    ]

    with pytest.raises(IndexError):
        extractor.extract_structures(
            representatives=reps,
            trajectories=[traj],
            output_dir=str(tmp_path / "out"),
            prefix="state",
            topology_path=None,
            trajectory_locator=None,
        )


def test_in_memory_empty_representatives_returns_empty_list(tmp_path: Path, extractor: RepresentativePicker) -> None:
    """Empty representatives should produce no files but still succeed."""

    traj = FakeTrajectory(n_frames=3)
    reps: List[RepresentativeFrame] = []

    out_dir = tmp_path / "empty"
    result = extractor.extract_structures(
        representatives=reps,
        trajectories=[traj],
        output_dir=str(out_dir),
        prefix="state",
        topology_path=None,
        trajectory_locator=None,
    )

    assert result == []
    assert out_dir.exists()
    assert list(out_dir.iterdir()) == []


def test_raw_mode_requires_topology_path(tmp_path: Path, extractor: RepresentativePicker) -> None:
    """Trajectory locator without topology path is a logical error."""

    traj_path = tmp_path / "traj.xtc"
    traj_path.write_text("dummy")

    locator = FakeTrajectoryLocator(traj_path)
    reps: List[RepresentativeFrame] = [
        (0, 10, None, None),
    ]

    with pytest.raises(ValueError):
        extractor.extract_structures(
            representatives=reps,
            trajectories=None,
            output_dir=str(tmp_path / "out"),
            prefix="state",
            topology_path=None,
            trajectory_locator=locator,
        )


def test_raw_mode_basic_saves_files(tmp_path: Path, extractor: RepresentativePicker, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw mode writes files via mdtraj using the locator and topology."""

    traj_path = tmp_path / "traj.xtc"
    traj_path.write_text("dummy trajectory")

    top_path = tmp_path / "topology.pdb"
    top_path.write_text("dummy topology")

    locator = FakeTrajectoryLocator(traj_path, fixed_index=0)
    reps: List[RepresentativeFrame] = [
        (2, 42, None, None),
        (3, 99, None, None),
    ]

    def fake_load_frame(path: str, index: int, top: str) -> FakeFrame:
        assert Path(path) == traj_path
        assert Path(top) == top_path
        assert isinstance(index, int)
        return FakeFrame()

    fake_mdtraj = SimpleNamespace(load_frame=fake_load_frame)
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)

    out_dir = tmp_path / "raw_mode_out"
    result = extractor.extract_structures(
        representatives=reps,
        trajectories=None,
        output_dir=str(out_dir),
        prefix="raw",
        topology_path=str(top_path),
        trajectory_locator=locator,
    )

    assert len(result) == 2
    for path, (state_id, global_idx, _, _) in zip(result, reps):
        saved = Path(path)
        assert saved.exists()
        assert saved.name == f"raw_{state_id:03d}_{global_idx:06d}.pdb"

    assert locator.calls == [42, 99]


def test_raw_mode_raises_if_trajectory_file_missing(tmp_path: Path, extractor: RepresentativePicker, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing trajectory files should surface as FileNotFoundError."""

    traj_path = tmp_path / "missing.xtc"
    top_path = tmp_path / "topology.pdb"
    top_path.write_text("dummy topology")

    locator = FakeTrajectoryLocator(traj_path, fixed_index=0)
    reps: List[RepresentativeFrame] = [
        (0, 10, None, None),
    ]

    fake_mdtraj = SimpleNamespace(load_frame=lambda *args, **kwargs: FakeFrame())
    monkeypatch.setitem(sys.modules, "mdtraj", fake_mdtraj)

    with pytest.raises(FileNotFoundError):
        extractor.extract_structures(
            representatives=reps,
            trajectories=None,
            output_dir=str(tmp_path / "out"),
            prefix="raw",
            topology_path=str(top_path),
            trajectory_locator=locator,
        )
