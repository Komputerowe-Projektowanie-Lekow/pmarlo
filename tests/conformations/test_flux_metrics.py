"""Tests for reactive flux utilities in the conformations module."""

from __future__ import annotations

import numpy as np

from pmarlo.conformations.finder import _calculate_state_flux
from pmarlo.conformations.tpt_analysis import TPTAnalysis


def test_calculate_state_flux_averages_incoming_and_outgoing() -> None:
    """Reactive flux through a state should average incoming and outgoing flow."""

    flux_matrix = np.array([[0.0, 4.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    # Incoming + outgoing, divided by two to avoid double counting each edge.
    expected = np.array([2.5, 2.0, 1.0])

    np.testing.assert_allclose(_calculate_state_flux(flux_matrix), expected)


def test_find_bottleneck_states_orders_by_average_flux() -> None:
    """Bottleneck state ranking should follow the averaged reactive flux."""

    flux_matrix = np.array([[0.0, 3.0, 1.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0]])
    expected_order = np.argsort(_calculate_state_flux(flux_matrix))[::-1]

    analysis = object.__new__(TPTAnalysis)
    ranked = TPTAnalysis.find_bottleneck_states(analysis, flux_matrix, top_n=3)

    np.testing.assert_array_equal(ranked, expected_order)
