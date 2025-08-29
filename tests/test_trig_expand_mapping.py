import numpy as np

from pmarlo.api import _trig_expand_periodic


def test_trig_expand_returns_mapping() -> None:
    X = np.array([[0.0, 0.1, 0.2], [1.0, 0.3, 0.4]], dtype=float)
    periodic = np.array([False, True, True])
    Xe, mapping = _trig_expand_periodic(X, periodic)
    assert Xe.shape == (2, 5)
    assert mapping.tolist() == [0, 1, 1, 2, 2]
    # Verify columns align with cos/sin expansion
    assert np.allclose(Xe[:, 0], X[:, 0])
    assert np.allclose(Xe[:, 1], np.cos(X[:, 1]))
    assert np.allclose(Xe[:, 2], np.sin(X[:, 1]))
    assert np.allclose(Xe[:, 3], np.cos(X[:, 2]))
    assert np.allclose(Xe[:, 4], np.sin(X[:, 2]))
