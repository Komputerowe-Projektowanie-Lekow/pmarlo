from __future__ import annotations

import numpy as np
import pytest

from pmarlo.features.deeptica.core.inputs import FeaturePrep, prepare_features


def test_prepare_features_concatenates_and_scales():
    blocks = [
        np.ones((4, 2), dtype=np.float32),
        np.arange(6, dtype=np.float32).reshape(3, 2),
    ]
    prep = prepare_features(blocks, tau_schedule=(2, 5), seed=42)
    assert isinstance(prep, FeaturePrep)
    assert prep.X.shape == (7, 2)
    assert prep.Z.shape == (7, 2)
    assert prep.tau_schedule == (2, 5)
    assert prep.input_dim == 2


def test_prepare_features_rejects_empty_schedule():
    with pytest.raises(ValueError):
        prepare_features([np.zeros((4, 1))], tau_schedule=(), seed=1)
