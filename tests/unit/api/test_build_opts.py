from __future__ import annotations

import pytest

from pmarlo.api import _build_opts


def test_build_opts_honours_microstate_request() -> None:
    opts = _build_opts(
        seed=42,
        temperature=310.0,
        lag=5,
        kmeans_kwargs={"n_init": 7},
        n_microstates=17,
    )

    assert opts.n_states == 17
    assert opts.n_clusters == 17
    assert opts.kmeans_kwargs == {"n_init": 7}


def test_build_opts_rejects_non_positive_microstates() -> None:
    with pytest.raises(ValueError):
        _build_opts(
            seed=1,
            temperature=300.0,
            lag=2,
            kmeans_kwargs=None,
            n_microstates=0,
        )
