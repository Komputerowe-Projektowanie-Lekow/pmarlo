import numpy as np
import pytest

from pmarlo.api import generate_fes_and_pick_minima
from pmarlo.msm.fes import FESResult


def test_fes_result_attribute_and_dict_access():
    F = np.array([[1.0, 2.0], [0.5, 3.0]])
    xedges = np.array([0.0, 1.0, 2.0])
    yedges = np.array([0.0, 1.0, 2.0])
    fes = FESResult(F=F, xedges=xedges, yedges=yedges)

    # Attribute access works
    assert fes.F is F
    assert fes.xedges[0] == 0.0

    # Dict-style access issues a warning but still works
    with pytest.warns(DeprecationWarning):
        assert np.array_equal(fes["F"], F)

    # API function runs without raising
    coords, result = generate_fes_and_pick_minima(fes)
    assert coords == (1.0, 0.0)  # minimum at F[1,0]
    assert result is fes
