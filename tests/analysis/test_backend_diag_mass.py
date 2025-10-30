from __future__ import annotations

import numpy as np
import pytest

from pmarlo_webapp.app.backend import _compute_analysis_diag_mass


def test_compute_analysis_diag_mass_from_stochastic_matrix():
    transition_matrix = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.05, 0.15, 0.8],
        ],
        dtype=np.float64,
    )

    diag_mass, matrix, guardrail = _compute_analysis_diag_mass(transition_matrix)

    expected = float(np.nanmean(np.diag(transition_matrix)))
    assert guardrail is None
    assert matrix is not None
    assert diag_mass == pytest.approx(expected)


@pytest.mark.parametrize(
    "transition",
    [None, np.zeros((0, 0), dtype=np.float64)],
)
def test_compute_analysis_diag_mass_guardrail_for_missing_matrix(transition):
    diag_mass, matrix, guardrail = _compute_analysis_diag_mass(transition)

    assert guardrail is not None
    assert guardrail["code"] == "diag_mass_unavailable"
    assert not np.isfinite(diag_mass)
    assert matrix is None

    guardrails = [guardrail]
    analysis_healthy = not guardrails
    assert not analysis_healthy
