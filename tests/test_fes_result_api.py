import numpy as np
import pytest

from pmarlo.api import generate_fes_and_pick_minima
from pmarlo.fes import FESResult


def test_fesresult_attribute_and_mapping_access():
    fes = FESResult(
        F=np.zeros((1, 1)),
        xedges=np.array([0.0, 1.0]),
        yedges=np.array([0.0, 1.0]),
    )
    assert fes.output_shape == (1, 1)
    with pytest.warns(DeprecationWarning):
        assert np.array_equal(fes["F"], fes.F)


def test_generate_fes_and_pick_minima_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    cols = ["a", "b"]
    periodic = np.array([False, False])
    res = generate_fes_and_pick_minima(
        X,
        cols,
        periodic,
        requested_pair=("a", "b"),
        bins=(10, 10),
        temperature=300.0,
        smooth=True,
        min_count=1,
        kde_bw_deg=(20.0, 20.0),
        deltaF_kJmol=1.0,
    )
    fes = res["fes"]
    assert isinstance(fes, FESResult)
    assert len(fes.output_shape) == 2
