import numpy as np
from sklearn.datasets import make_blobs

from pmarlo.cluster.micro import cluster_microstates
from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM
from pmarlo.utils.seed import set_global_seed


def _transition_from_labels(labels: np.ndarray, out_dir: str) -> np.ndarray:
    """Build a transition matrix for testing purposes."""
    msm = EnhancedMSM(random_state=None, output_dir=out_dir)
    msm.n_states = int(labels.max()) + 1 if labels.size else 0
    msm.dtrajs = [labels]
    msm._build_standard_msm(lag_time=1)
    assert msm.transition_matrix is not None
    return np.asarray(msm.transition_matrix, dtype=float)


def test_reproducible_clustering_and_msm(tmp_path):
    """Two runs with the same seed should be identical."""
    data, _ = make_blobs(n_samples=200, centers=4, n_features=2, random_state=0)

    set_global_seed(123)
    res1 = cluster_microstates(data, n_states=4, random_state=None)
    T1 = _transition_from_labels(res1.labels, str(tmp_path / "run1"))

    set_global_seed(123)
    res2 = cluster_microstates(data, n_states=4, random_state=None)
    T2 = _transition_from_labels(res2.labels, str(tmp_path / "run2"))

    assert np.array_equal(res1.labels, res2.labels)
    assert np.allclose(T1, T2)
