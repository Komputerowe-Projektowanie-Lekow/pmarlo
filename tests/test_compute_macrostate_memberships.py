import numpy as np
import pytest

pytest.importorskip("deeptime")

from pmarlo.conformations.finder import _compute_macrostate_memberships


def test_shapes_and_label_consistency_simple_chain():
    T = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    pi = np.array([0.5, 0.5], dtype=float)
    n_metastable = 2

    memberships, labels = _compute_macrostate_memberships(T, pi, n_metastable)

    n_states = T.shape[0]
    assert memberships.shape == (n_states, n_metastable)
    assert labels.shape == (n_states,)

    for i in range(n_states):
        assert 0 <= labels[i] < n_metastable
        assert labels[i] == int(np.argmax(memberships[i]))


def test_macrostate_ordering_by_population_weight():
    T = np.array(
        [
            [0.95, 0.05, 0.0],
            [0.05, 0.9, 0.05],
            [0.0, 0.1, 0.9],
        ],
        dtype=float,
    )
    pi = np.array([0.7, 0.2, 0.1], dtype=float)
    n_metastable = 2

    memberships, _ = _compute_macrostate_memberships(T, pi, n_metastable)

    macro_weights = pi @ memberships
    assert macro_weights.shape == (n_metastable,)
    assert np.all(macro_weights[:-1] >= macro_weights[1:])


def test_membership_rows_look_like_probabilities():
    T = np.array(
        [
            [0.8, 0.2],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    pi = np.array([0.6, 0.4], dtype=float)
    n_metastable = 2

    memberships, _ = _compute_macrostate_memberships(T, pi, n_metastable)

    row_sums = memberships.sum(axis=1)
    assert np.all(row_sums == pytest.approx(1.0, rel=1e-6, abs=1e-6))
    assert np.all(memberships >= -1e-12)


def test_single_macrostate_degenerates_to_one_column():
    T = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    pi = np.array([0.5, 0.5], dtype=float)
    n_metastable = 1

    memberships, labels = _compute_macrostate_memberships(T, pi, n_metastable)

    n_states = T.shape[0]
    assert memberships.shape == (n_states, 1)
    assert labels.shape == (n_states,)
    assert np.all(labels == 0)

    row_sums = memberships.sum(axis=1)
    assert np.all(row_sums == pytest.approx(1.0, rel=1e-6, abs=1e-6))


def test_non_square_transition_matrix_raises():
    T = np.array([[0.9, 0.1, 0.0]], dtype=float)
    pi = np.array([1.0], dtype=float)

    with pytest.raises(ValueError):
        _compute_macrostate_memberships(T, pi, n_metastable=1)


def test_n_metastable_too_large_raises():
    T = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    pi = np.array([0.5, 0.5], dtype=float)

    with pytest.raises(ValueError):
        _compute_macrostate_memberships(T, pi, n_metastable=3)


def test_n_metastable_less_than_one_raises():
    T = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    pi = np.array([0.5, 0.5], dtype=float)

    with pytest.raises(ValueError):
        _compute_macrostate_memberships(T, pi, n_metastable=0)


def test_pi_length_mismatch_raises():
    T = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    pi = np.array([1.0, 0.0, 0.0], dtype=float)

    with pytest.raises(ValueError):
        _compute_macrostate_memberships(T, pi, n_metastable=2)
