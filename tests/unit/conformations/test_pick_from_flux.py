from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.representative_picker import RepresentativePicker


class TestablePicker(RepresentativePicker):
    """Subclass that reuses pick_from_flux but spies on pick_representatives."""

    def __init__(self) -> None:
        # pick_from_flux only uses pick_representatives, so no need to call super().
        self.last_call = None

    def pick_representatives(
        self,
        features,
        dtrajs,
        states,
        weights=None,
        n_reps=1,
        method="medoid",
    ):
        """Spy implementation that records inputs and returns dummy data."""
        self.last_call = {
            "features": features,
            "dtrajs": dtrajs,
            "states": np.array(states, copy=True),
            "weights": weights,
            "n_reps": n_reps,
            "method": method,
        }

        reps = []
        for state in states:
            for rep_idx in range(n_reps):
                reps.append(("rep", int(state), rep_idx))
        return reps


def _dummy_features_and_dtrajs(n_states: int):
    """Minimal consistent inputs for the picker."""
    dtrajs = [np.arange(n_states, dtype=int)]
    features = np.zeros((n_states, 2), dtype=float)
    return features, dtrajs


def test_pick_from_flux_prefers_high_flux_states():
    """
    For a small example, states with largest through-flux should be selected.
    """
    picker = TestablePicker()

    flux_matrix = np.array(
        [
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    features, dtrajs = _dummy_features_and_dtrajs(n_states=4)

    result = picker.pick_from_flux(
        flux_matrix=flux_matrix,
        features=features,
        dtrajs=dtrajs,
        top_n=2,
        n_reps_per_state=3,
        weights=None,
    )

    expected_states = np.array([1, 0], dtype=int)

    np.testing.assert_array_equal(picker.last_call["states"], expected_states)
    assert picker.last_call["n_reps"] == 3
    assert picker.last_call["method"] == "medoid"

    assert result == [
        ("rep", 1, 0),
        ("rep", 1, 1),
        ("rep", 1, 2),
        ("rep", 0, 0),
        ("rep", 0, 1),
        ("rep", 0, 2),
    ]


def test_pick_from_flux_invariant_to_global_flux_rescaling():
    """
    Multiplying the flux matrix by a positive constant should not change the ranking.
    """
    picker1 = TestablePicker()
    picker2 = TestablePicker()

    base_flux = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
        dtype=float,
    )

    features, dtrajs = _dummy_features_and_dtrajs(n_states=3)

    picker1.pick_from_flux(
        flux_matrix=base_flux,
        features=features,
        dtrajs=dtrajs,
        top_n=3,
        n_reps_per_state=1,
    )
    picker2.pick_from_flux(
        flux_matrix=10.0 * base_flux,
        features=features,
        dtrajs=dtrajs,
        top_n=3,
        n_reps_per_state=1,
    )

    np.testing.assert_array_equal(
        picker1.last_call["states"], picker2.last_call["states"]
    )


def test_pick_from_flux_clamps_top_n_to_number_of_states():
    """If top_n is larger than available MSM states, all states should be used."""
    picker = TestablePicker()

    n_states = 3
    flux_matrix = np.eye(n_states, dtype=float)
    features, dtrajs = _dummy_features_and_dtrajs(n_states=n_states)

    picker.pick_from_flux(
        flux_matrix=flux_matrix,
        features=features,
        dtrajs=dtrajs,
        top_n=10,
        n_reps_per_state=1,
    )

    states = picker.last_call["states"]
    assert len(states) == n_states
    assert set(states.tolist()) == {0, 1, 2}


def test_pick_from_flux_requires_square_flux_matrix():
    """The flux matrix must be square; other shapes should raise errors."""
    picker = TestablePicker()

    flux_matrix = np.ones((3, 4), dtype=float)
    features, dtrajs = _dummy_features_and_dtrajs(n_states=3)

    with pytest.raises(ValueError):
        picker.pick_from_flux(
            flux_matrix=flux_matrix,
            features=features,
            dtrajs=dtrajs,
        )


@pytest.mark.parametrize("top_n", [0, -1])
def test_pick_from_flux_rejects_non_positive_top_n(top_n):
    """Requesting zero or negative numbers of bottleneck states should error."""
    picker = TestablePicker()

    flux_matrix = np.eye(2, dtype=float)
    features, dtrajs = _dummy_features_and_dtrajs(n_states=2)

    with pytest.raises(ValueError):
        picker.pick_from_flux(
            flux_matrix=flux_matrix,
            features=features,
            dtrajs=dtrajs,
            top_n=top_n,
        )


@pytest.mark.parametrize("n_reps_per_state", [0, -2])
def test_pick_from_flux_rejects_non_positive_n_reps(n_reps_per_state):
    """n_reps_per_state must be at least 1."""
    picker = TestablePicker()

    flux_matrix = np.eye(2, dtype=float)
    features, dtrajs = _dummy_features_and_dtrajs(n_states=2)

    with pytest.raises(ValueError):
        picker.pick_from_flux(
            flux_matrix=flux_matrix,
            features=features,
            dtrajs=dtrajs,
            n_reps_per_state=n_reps_per_state,
        )
