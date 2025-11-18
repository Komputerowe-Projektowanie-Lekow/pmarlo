"""Behavioral tests for `_update_with_representatives`."""

from __future__ import annotations

import pytest

from pmarlo.conformations.finder import _update_with_representatives
from pmarlo.conformations.results import Conformation


def make_conf(state_id: int, frame: int = -1, traj: int | None = None, local: int | None = None) -> Conformation:
    """Helper to build minimally valid `Conformation` instances."""

    return Conformation(
        conformation_type="metastable",
        state_id=state_id,
        frame_index=frame,
        population=1.0,
        free_energy=0.0,
        trajectory_index=traj,
        local_frame_index=local,
    )


def snapshot(conformations: list[Conformation]) -> list[tuple[int, int | None, int | None]]:
    """Capture the indices we mutate so tests can assert no partial updates."""

    return [
        (c.frame_index, c.trajectory_index, c.local_frame_index) for c in conformations
    ]


def test_updates_conformations_with_matching_state_ids() -> None:
    """Frame metadata must match exactly what representatives specify."""

    conformations = [
        make_conf(0, frame=0, traj=0, local=0),
        make_conf(1, frame=1, traj=1, local=1),
        make_conf(2, frame=2, traj=2, local=2),
    ]

    representatives = [
        (0, 10, 100, 1000),
        (2, 20, 200, 2000),
    ]

    _update_with_representatives(conformations, representatives)

    assert conformations[0].frame_index == 10
    assert conformations[0].trajectory_index == 100
    assert conformations[0].local_frame_index == 1000

    assert conformations[1].frame_index == 1
    assert conformations[1].trajectory_index == 1
    assert conformations[1].local_frame_index == 1

    assert conformations[2].frame_index == 20
    assert conformations[2].trajectory_index == 200
    assert conformations[2].local_frame_index == 2000


def test_conformations_without_representative_are_unchanged() -> None:
    """States without a representative must remain untouched."""

    conformations = [
        make_conf(0, frame=5, traj=6, local=7),
        make_conf(1, frame=8, traj=9, local=10),
    ]

    representatives = [
        (0, 100, 200, 300),
    ]

    _update_with_representatives(conformations, representatives)

    assert conformations[0].frame_index == 100
    assert conformations[0].trajectory_index == 200
    assert conformations[0].local_frame_index == 300

    assert conformations[1].frame_index == 8
    assert conformations[1].trajectory_index == 9
    assert conformations[1].local_frame_index == 10


def test_empty_inputs_do_not_crash() -> None:
    """Empty collections should be accepted."""

    conformations: list[Conformation] = []
    representatives: list[tuple[int, int, int, int]] = []

    _update_with_representatives(conformations, representatives)

    assert conformations == []


def test_raises_on_duplicate_state_ids_and_does_not_partially_modify() -> None:
    """Duplicates should fail and leave all conformations unchanged."""

    conformations = [
        make_conf(0, frame=1, traj=2, local=3),
        make_conf(1, frame=4, traj=5, local=6),
    ]

    representatives = [
        (0, 10, 20, 30),
        (0, 11, 21, 31),
    ]

    before = snapshot(conformations)

    with pytest.raises(ValueError):
        _update_with_representatives(conformations, representatives)

    assert snapshot(conformations) == before


def test_raises_on_representative_for_unknown_state_and_does_not_partially_modify() -> None:
    """Unknown state references must raise and leave the input untouched."""

    conformations = [
        make_conf(0, frame=1, traj=2, local=3),
        make_conf(1, frame=4, traj=5, local=6),
    ]

    representatives = [
        (0, 10, 20, 30),
        (2, 40, 50, 60),
    ]

    before = snapshot(conformations)

    with pytest.raises(ValueError):
        _update_with_representatives(conformations, representatives)

    assert snapshot(conformations) == before
