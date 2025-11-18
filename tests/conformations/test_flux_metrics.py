"""Tests for reactive flux utilities in the conformations module."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt

from pmarlo.conformations.finder import _calculate_state_flux
from pmarlo.conformations.tpt_analysis import TPTAnalysis


def test_single_edge_two_states() -> None:
    """Single directed edge shares the same mean flux between its two states."""
    flux = np.array([[0.0, 2.0], [0.0, 0.0]], dtype=float)
    expected = np.array([2.0, 2.0], dtype=float)

    result = _calculate_state_flux(flux)
    npt.assert_allclose(result, expected)


def test_isolated_state_has_zero_flux() -> None:
    """States not touching any nonzero edge have zero mean flux."""
    flux = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=float,
    )
    expected = np.array([1.0, 1.0, 0.0], dtype=float)

    result = _calculate_state_flux(flux)
    npt.assert_allclose(result, expected)


def test_hub_with_many_weak_edges_not_dominant() -> None:
    """Few strong edges should dominate over many weak edges."""
    flux = np.array(
        [[0.0, 0.1, 0.1], [0.1, 0.0, 1.0], [0.1, 1.0, 0.0]],
        dtype=float,
    )
    expected = np.array([0.1, 0.55, 0.55], dtype=float)

    result = _calculate_state_flux(flux)
    npt.assert_allclose(result, expected)
    assert result[1] > result[0]
    assert result[2] > result[0]


def test_transpose_invariance() -> None:
    """Flux through a state does not depend on swapping incoming and outgoing."""
    rng = np.random.default_rng(123)
    flux = rng.random((5, 5))
    mask = rng.random((5, 5)) < 0.4
    flux[mask] = 0.0

    result = _calculate_state_flux(flux)
    result_T = _calculate_state_flux(flux.T)

    npt.assert_allclose(result, result_T)


def test_scaling_is_linear() -> None:
    """Scaling all fluxes by a constant scales the scores by the same constant."""
    flux = np.array(
        [[0.0, 0.2, 0.0], [0.0, 0.0, 0.5], [0.3, 0.0, 0.0]],
        dtype=float,
    )
    alpha = 3.7

    base = _calculate_state_flux(flux)
    scaled = _calculate_state_flux(alpha * flux)

    npt.assert_allclose(scaled, alpha * base)


def test_find_bottleneck_states_orders_by_average_flux() -> None:
    """Bottleneck state ranking should follow the averaged reactive flux."""

    flux_matrix = np.array([[0.0, 3.0, 1.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]])
    expected_order = np.argsort(_calculate_state_flux(flux_matrix))[::-1]

    analysis = object.__new__(TPTAnalysis)
    ranked = TPTAnalysis.find_bottleneck_states(analysis, flux_matrix, top_n=3)

    npt.assert_array_equal(ranked, expected_order)
