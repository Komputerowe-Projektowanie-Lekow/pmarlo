from __future__ import annotations

import numpy as np

from pmarlo.cluster.micro import cluster_microstates
from pmarlo.states.msm_bridge import _fit_msm_fallback
from pmarlo.utils.seed import set_global_seed


def _run(seed: int) -> tuple[np.ndarray, np.ndarray]:
    set_global_seed(seed)
    Y = np.random.rand(100, 3)
    labels = cluster_microstates(Y, n_clusters=5, random_state=seed)
    T, _ = _fit_msm_fallback([labels], n_states=5, lag=1)
    return labels, T


def test_reproducible_clustering_and_msm() -> None:
    labels1, T1 = _run(123)
    labels2, T2 = _run(123)
    assert np.array_equal(labels1, labels2)
    assert np.allclose(T1, T2)
