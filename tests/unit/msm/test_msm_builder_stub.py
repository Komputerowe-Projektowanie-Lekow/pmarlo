from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmarlo.markov_state_model.msm_builder import MSMBuilder


def test_msm_builder_placeholder():
    builder = MSMBuilder(tau_steps=2, n_clusters=3, random_state=42)
    Y_list = [np.random.rand(5, 2), np.random.rand(7, 2)]
    result = builder.fit(Y_list)
    assert result.T.shape == (3, 3)
    assert result.pi.shape == (3,)
    assert np.isclose(result.pi.sum(), 1.0)
    assert result.its.shape == (2,)
    assert result.clusters.shape[0] == sum(arr.shape[0] for arr in Y_list)
