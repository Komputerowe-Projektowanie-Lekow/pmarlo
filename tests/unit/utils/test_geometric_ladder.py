from __future__ import annotations

import numpy as np

from pmarlo.utils.replica_utils import geometric_ladder


def test_geometric_ladder_monotone_and_endpoints():
    arr = geometric_ladder(300.0, 600.0, 5)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 5
    assert np.all(np.diff(arr) > 0)
    assert abs(arr[0] - 300.0) < 1e-9
    assert abs(arr[-1] - 600.0) < 1e-9
    ratios = arr[1:] / arr[:-1]
    assert np.allclose(ratios, np.full_like(ratios, ratios[0]))

def test_geometric_ladder_invalid():
    import pytest

    with pytest.raises(ValueError):
        geometric_ladder(300.0, 300.0, 5)
    with pytest.raises(ValueError):
        geometric_ladder(0.0, 300.0, 5)
    with pytest.raises(ValueError):
        geometric_ladder(300.0, 600.0, 1)

