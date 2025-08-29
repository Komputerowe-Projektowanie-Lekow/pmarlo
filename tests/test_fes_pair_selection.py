import numpy as np

from pmarlo.api import select_fes_pair


def test_select_fes_pair_ignores_constant_axes():
    # First column constant, others varying
    X = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    cols = ["c0", "c1", "c2"]
    periodic = np.array([False, False, False])
    i, j, pi, pj = select_fes_pair(X, cols, periodic)
    assert {i, j} == {1, 2}
    assert not pi and not pj


def test_select_fes_pair_all_constant_folds():
    X = np.ones((3, 2))
    cols = ["c0", "c1"]
    periodic = np.array([False, False])
    i, j, pi, pj = select_fes_pair(X, cols, periodic)
    assert (i, j) == (0, 0)
    assert not pi and not pj
