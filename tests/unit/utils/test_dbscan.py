import numpy as np

from pmarlo.utils.dbscan import estimate_dbscan_eps


def _two_cluster_data(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=0.0, scale=0.15, size=(200, 3))
    cluster_b = rng.normal(loc=3.0, scale=0.2, size=(200, 3))
    return np.vstack([cluster_a, cluster_b])


def test_estimate_dbscan_eps_returns_positive_radius() -> None:
    data = _two_cluster_data()
    eps, meta = estimate_dbscan_eps(data, min_samples=5, random_state=123)
    assert eps > 0
    assert meta["sample_size"] <= data.shape[0]
    assert meta["neighbor_rank"] >= 2


def test_estimate_dbscan_eps_is_deterministic_with_seed() -> None:
    data = _two_cluster_data()
    eps1, _ = estimate_dbscan_eps(data, min_samples=8, random_state=42)
    eps2, _ = estimate_dbscan_eps(data, min_samples=8, random_state=42)
    assert eps1 == eps2
