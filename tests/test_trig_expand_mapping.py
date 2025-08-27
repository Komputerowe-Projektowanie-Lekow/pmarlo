import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pmarlo.api import _trig_expand_periodic


def test_trig_expand_returns_mapping():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    periodic = np.array([True, False])
    Xe, mapping = _trig_expand_periodic(X, periodic)
    assert Xe.shape == (2, 3)
    assert mapping == {0: (0, 1), 1: (2,)}
    expected = np.column_stack([np.cos(X[:, 0]), np.sin(X[:, 0]), X[:, 1]])
    assert np.allclose(Xe, expected)
