from __future__ import annotations

import numpy as np

from pmarlo.markov_state_model._msm_utils import _expand_results as utils_expand
from pmarlo.markov_state_model.bridge import _expand_results as bridge_expand


def test_expand_results_handles_underestimated_state_count() -> None:
    """Embedding should grow to accommodate estimator-provided indices."""

    active = np.array([0, 3], dtype=int)
    T_active = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    pi_active = np.array([0.6, 0.4], dtype=float)

    T_full_utils, pi_full_utils = utils_expand(2, active, T_active, pi_active)
    assert T_full_utils.shape == (4, 4)
    np.testing.assert_allclose(T_full_utils[np.ix_(active, active)], T_active)
    np.testing.assert_allclose(pi_full_utils[active], pi_active)
    # Unused slots remain self-transitions to maintain stochasticity.
    np.testing.assert_allclose(np.diag(T_full_utils)[1:3], np.ones(2))

    T_full_bridge, pi_full_bridge = bridge_expand(2, active, T_active, pi_active)
    np.testing.assert_array_equal(T_full_bridge, T_full_utils)
    np.testing.assert_array_equal(pi_full_bridge, pi_full_utils)
