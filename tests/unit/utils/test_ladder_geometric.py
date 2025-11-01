from __future__ import annotations

import numpy as np

from pmarlo.utils.replica_utils import geometric_temperature_ladder


def test_geometric_ladder_sorted_and_bounds():
    temps = geometric_temperature_ladder(300.0, 600.0, 5)
    assert len(temps) == 5
    # sorted ascending and includes bounds
    assert temps == sorted(temps)
    assert abs(temps[0] - 300.0) < 1e-9
    assert abs(temps[-1] - 600.0) < 1e-9
    # geometric spacing ratios are (approximately) constant
    ratios = [temps[i + 1] / temps[i] for i in range(len(temps) - 1)]
    assert np.allclose(ratios, np.full(len(ratios), ratios[0]))
