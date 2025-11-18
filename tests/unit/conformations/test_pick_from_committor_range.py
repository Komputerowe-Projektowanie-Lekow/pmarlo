"""Behavioral tests for pick_from_committor_range."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.representative_picker import RepresentativePicker


class DummyPicker:
    """Stub picker that records delegated calls for inspection."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def pick_representatives(
        self,
        features,
        dtrajs,
        states,
        weights=None,
        n_reps: int = 1,
        method: str = "medoid",
    ):
        """Record how pick_from_committor_range calls the generic picker."""

        self.calls.append(
            {
                "features": np.asarray(features),
                "dtrajs": [np.asarray(dt) for dt in dtrajs],
                "states": np.asarray(states),
                "weights": None if weights is None else np.asarray(weights),
                "n_reps": n_reps,
                "method": method,
            }
        )
        return ["dummy-result"]


DummyPicker.pick_from_committor_range = RepresentativePicker.pick_from_committor_range


def test_pick_from_committor_range_selects_states_in_range() -> None:
    picker = DummyPicker()

    committor = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    features = np.zeros((5, 2))
    dtrajs = [np.array([0, 1, 2, 3, 4])]

    result = picker.pick_from_committor_range(
        committor=committor,
        features=features,
        dtrajs=dtrajs,
        committor_range=(0.4, 0.6),
        n_reps=2,
    )

    assert len(picker.calls) == 1
    call = picker.calls[0]

    assert set(call["states"].tolist()) == {2}
    assert call["n_reps"] == 2
    assert call["method"] == "diverse"
    assert result == ["dummy-result"]


def test_pick_from_committor_range_multiple_states_in_range() -> None:
    picker = DummyPicker()

    committor = np.array([0.45, 0.55, 0.2, 0.8])
    features = np.zeros((4, 3))
    dtrajs = [np.array([0, 1, 2, 3])]

    picker.pick_from_committor_range(
        committor=committor,
        features=features,
        dtrajs=dtrajs,
        committor_range=(0.4, 0.6),
        n_reps=3,
    )

    call = picker.calls[0]

    assert set(call["states"].tolist()) == {0, 1}
    assert call["n_reps"] == 3
    assert call["method"] == "diverse"


def test_pick_from_committor_range_includes_boundaries() -> None:
    picker = DummyPicker()

    committor = np.array([0.4, 0.5, 0.6, 0.39, 0.61])
    features = np.zeros((5, 1))
    dtrajs = [np.array([0, 1, 2, 3, 4])]

    picker.pick_from_committor_range(
        committor=committor,
        features=features,
        dtrajs=dtrajs,
        committor_range=(0.4, 0.6),
        n_reps=1,
    )

    call = picker.calls[0]

    assert set(call["states"].tolist()) == {0, 1, 2}


def test_pick_from_committor_range_raises_if_no_states_found() -> None:
    picker = DummyPicker()

    committor = np.array([0.0, 0.1, 0.2])
    features = np.zeros((3, 1))
    dtrajs = [np.array([0, 1, 2])]

    with pytest.raises(ValueError, match="No states found in committor range"):
        picker.pick_from_committor_range(
            committor=committor,
            features=features,
            dtrajs=dtrajs,
            committor_range=(0.4, 0.6),
            n_reps=1,
        )


def test_pick_from_committor_range_rejects_invalid_range_order() -> None:
    picker = DummyPicker()

    committor = np.array([0.2, 0.5, 0.8])
    features = np.zeros((3, 1))
    dtrajs = [np.array([0, 1, 2])]

    with pytest.raises(ValueError):
        picker.pick_from_committor_range(
            committor=committor,
            features=features,
            dtrajs=dtrajs,
            committor_range=(0.7, 0.3),
            n_reps=1,
        )


def test_pick_from_committor_range_rejects_range_outside_01() -> None:
    picker = DummyPicker()

    committor = np.array([0.2, 0.5, 0.8])
    features = np.zeros((3, 1))
    dtrajs = [np.array([0, 1, 2])]

    with pytest.raises(ValueError):
        picker.pick_from_committor_range(
            committor=committor,
            features=features,
            dtrajs=dtrajs,
            committor_range=(-0.1, 0.6),
            n_reps=1,
        )

    with pytest.raises(ValueError):
        picker.pick_from_committor_range(
            committor=committor,
            features=features,
            dtrajs=dtrajs,
            committor_range=(0.2, 1.1),
            n_reps=1,
        )


def test_pick_from_committor_range_requires_1d_committor() -> None:
    picker = DummyPicker()

    committor = np.zeros((2, 2))
    features = np.zeros((4, 1))
    dtrajs = [np.array([0, 1, 2, 3])]

    with pytest.raises(ValueError):
        picker.pick_from_committor_range(
            committor=committor,
            features=features,
            dtrajs=dtrajs,
            committor_range=(0.4, 0.6),
            n_reps=1,
        )


def test_pick_from_committor_range_checks_dtrajs_bounds() -> None:
    picker = DummyPicker()

    committor = np.array([0.3, 0.5, 0.7])
    features = np.zeros((3, 1))
    dtrajs = [np.array([0, 1, 3])]

    with pytest.raises(ValueError):
        picker.pick_from_committor_range(
            committor=committor,
            features=features,
            dtrajs=dtrajs,
            committor_range=(0.4, 0.6),
            n_reps=1,
        )


def test_pick_from_committor_range_rejects_nonpositive_n_reps() -> None:
    picker = DummyPicker()

    committor = np.array([0.4, 0.5, 0.6])
    features = np.zeros((3, 1))
    dtrajs = [np.array([0, 1, 2])]

    with pytest.raises(ValueError):
        picker.pick_from_committor_range(
            committor=committor,
            features=features,
            dtrajs=dtrajs,
            committor_range=(0.4, 0.6),
            n_reps=0,
        )

    with pytest.raises(ValueError):
        picker.pick_from_committor_range(
            committor=committor,
            features=features,
            dtrajs=dtrajs,
            committor_range=(0.4, 0.6),
            n_reps=-3,
        )
