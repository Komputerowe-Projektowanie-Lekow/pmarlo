"""Behavioral tests for :func:`_extract_structures`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytest

import pmarlo.conformations.finder as finder
from pmarlo.conformations.results import Conformation


class DummyTrajectory:
    """Stand-in trajectories container."""


class DummyPicker:
    """Captures extract calls and returns fabricated file paths."""

    def __init__(self) -> None:
        self.calls: List[
            Tuple[
                List[Tuple[int, int, int, int]],
                Any,
                str,
                str,
                Optional[str],
                Optional[Any],
            ]
        ] = []

    def extract_structures(
        self,
        representatives: List[Tuple[int, int, int, int]],
        trajectories: Any,
        output_dir: str,
        prefix: str,
        *,
        topology_path: Optional[str] = None,
        trajectory_locator: Optional[Any] = None,
    ) -> List[str]:
        self.calls.append(
            (
                representatives,
                trajectories,
                output_dir,
                prefix,
                topology_path,
                trajectory_locator,
            )
        )

        fake_paths: List[str] = []
        base_dir = Path(output_dir)
        for state_id, frame_index, _traj_idx, _local_idx in representatives:
            filename = f"{prefix}_{state_id:03d}_{frame_index:06d}.pdb"
            fake_paths.append(str(base_dir / filename))
        return fake_paths


@pytest.fixture()
def monkeypatched_picker(monkeypatch: Any) -> DummyPicker:
    """Patch :class:`RepresentativePicker` to avoid touching real data."""

    dummy = DummyPicker()
    monkeypatch.setattr(finder, "RepresentativePicker", lambda: dummy)
    return dummy


def _make_conformation(
    *,
    state_id: int,
    conf_type: str = "metastable",
    frame_index: int = 0,
    trajectory_index: Optional[int] = None,
    local_frame_index: Optional[int] = None,
) -> Conformation:
    return Conformation(
        conformation_type=conf_type,
        state_id=state_id,
        frame_index=frame_index,
        population=0.1,
        free_energy=0.0,
        trajectory_index=trajectory_index,
        local_frame_index=local_frame_index,
    )


def test_basic_extraction_assigns_paths_per_type(monkeypatched_picker: DummyPicker) -> None:
    """Valid representatives of multiple types get per-type paths and picker invocations."""

    confs = [
        _make_conformation(
            state_id=0,
            frame_index=10,
            trajectory_index=0,
            local_frame_index=100,
        ),
        _make_conformation(
            state_id=1,
            frame_index=20,
            trajectory_index=0,
            local_frame_index=200,
        ),
        _make_conformation(
            state_id=2,
            conf_type="transition",
            frame_index=5,
            trajectory_index=1,
            local_frame_index=50,
        ),
        _make_conformation(
            state_id=3,
            conf_type="pathway",
            frame_index=7,
            trajectory_index=2,
            local_frame_index=70,
        ),
    ]

    trajectories = DummyTrajectory()
    output_dir = "out"

    finder._extract_structures(
        conformations=confs,
        trajectories=trajectories,
        output_dir=output_dir,
        topology_path="top.pdb",
        trajectory_locator=None,
    )

    assert confs[0].structure_path == str(
        Path(output_dir) / "metastable" / "metastable_000_000010.pdb"
    )
    assert confs[1].structure_path == str(
        Path(output_dir) / "metastable" / "metastable_001_000020.pdb"
    )
    assert confs[2].structure_path == str(
        Path(output_dir) / "transition" / "transition_002_000005.pdb"
    )
    assert confs[3].structure_path == str(
        Path(output_dir) / "pathway" / "pathway_003_000007.pdb"
    )

    assert len(monkeypatched_picker.calls) == 3
    prefixes = {call[3] for call in monkeypatched_picker.calls}
    assert prefixes == {"metastable", "transition", "pathway"}

    expected_reps_by_type = {}
    for conf_type in {"metastable", "transition", "pathway"}:
        reps: List[Tuple[int, int, int, int]] = []
        for conf in confs:
            if conf.conformation_type != conf_type or conf.frame_index < 0:
                continue
            assert conf.trajectory_index is not None
            assert conf.local_frame_index is not None
            reps.append(
                (
                    conf.state_id,
                    conf.frame_index,
                    conf.trajectory_index,
                    conf.local_frame_index,
                )
            )
        expected_reps_by_type[conf_type] = reps

    for reps, trajs, out_dir, prefix, topo, locator in monkeypatched_picker.calls:
        assert trajs is trajectories
        assert topo == "top.pdb"
        assert locator is None
        assert out_dir == str(Path(output_dir) / prefix)
        assert reps == expected_reps_by_type[prefix]


def test_negative_frame_index_is_skipped_without_shifting(monkeypatched_picker: DummyPicker) -> None:
    """States with invalid frame indices are ignored without affecting others."""

    confs = [
        _make_conformation(
            state_id=0,
            frame_index=-1,
            trajectory_index=0,
            local_frame_index=0,
        ),
        _make_conformation(
            state_id=1,
            frame_index=10,
            trajectory_index=0,
            local_frame_index=10,
        ),
        _make_conformation(
            state_id=2,
            frame_index=20,
            trajectory_index=0,
            local_frame_index=20,
        ),
    ]

    finder._extract_structures(
        conformations=confs,
        trajectories=DummyTrajectory(),
        output_dir="out",
    )

    assert confs[0].structure_path is None
    assert confs[1].structure_path and confs[1].structure_path.endswith(
        "metastable_001_000010.pdb"
    )
    assert confs[2].structure_path and confs[2].structure_path.endswith(
        "metastable_002_000020.pdb"
    )

    assert len(monkeypatched_picker.calls) == 1
    reps, *_ = monkeypatched_picker.calls[0]
    assert [state for state, *_ in reps] == [1, 2]
    assert [frame for _, frame, *_ in reps] == [10, 20]


def test_missing_trajectory_indices_raise_value_error(monkeypatched_picker: DummyPicker) -> None:
    """Missing trajectory or local indices raise and prevent extraction."""

    missing_traj = _make_conformation(
        state_id=0,
        frame_index=10,
        trajectory_index=None,
        local_frame_index=5,
    )

    with pytest.raises(ValueError):
        finder._extract_structures(
            conformations=[missing_traj],
            trajectories=DummyTrajectory(),
            output_dir="out",
        )

    assert len(monkeypatched_picker.calls) == 0

    missing_local = _make_conformation(
        state_id=1,
        frame_index=10,
        trajectory_index=0,
        local_frame_index=None,
    )

    with pytest.raises(ValueError):
        finder._extract_structures(
            conformations=[missing_local],
            trajectories=DummyTrajectory(),
            output_dir="out",
        )

    assert len(monkeypatched_picker.calls) == 0


def test_no_valid_representatives_means_no_picker_call(monkeypatched_picker: DummyPicker) -> None:
    """If no representatives exist for a type, picker is never invoked."""

    confs = [
        _make_conformation(
            state_id=0,
            frame_index=-1,
            trajectory_index=0,
            local_frame_index=0,
        ),
        _make_conformation(
            state_id=1,
            frame_index=-5,
            trajectory_index=0,
            local_frame_index=0,
        ),
    ]

    finder._extract_structures(
        conformations=confs,
        trajectories=DummyTrajectory(),
        output_dir="out",
    )

    assert all(conf.structure_path is None for conf in confs)
    assert len(monkeypatched_picker.calls) == 0
