import numpy as np

from pmarlo.analysis.fes import compute_weighted_fes


def _make_dataset(points: np.ndarray) -> dict:
    return {"splits": {"train": {"X": np.asarray(points, dtype=np.float64)}}}


def test_kde_surface_metadata_and_shape():
    rng = np.random.default_rng(4)
    cluster_a = rng.normal(loc=[-1.0, -0.5], scale=0.15, size=(80, 2))
    cluster_b = rng.normal(loc=[1.1, 0.7], scale=0.12, size=(70, 2))
    dataset = _make_dataset(np.vstack([cluster_a, cluster_b]))

    fes = compute_weighted_fes(dataset, method="kde", bins=16, bandwidth="scott")

    density = fes["histogram"]
    assert density.shape == (16, 16)
    assert np.all(np.isfinite(density))
    assert np.min(density) > 0.0

    metadata = fes["metadata"]
    assert metadata["method"] == "kde"
    assert metadata["bandwidth"]["selector"] == "scott"
    assert metadata["bandwidth"]["x"] > 0.0
    assert metadata["bandwidth"]["y"] > 0.0


def test_grid_smoothing_applies_neighbor_floor():
    points = np.array(
        [
            [0.02, 0.01],
            [0.03, -0.02],
            [-0.01, 0.04],
            [0.02, 0.03],
            [0.01, -0.01],
        ],
        dtype=np.float64,
    )
    dataset = _make_dataset(points)

    fes = compute_weighted_fes(
        dataset,
        method="grid",
        bins=(6, 6),
        min_count_per_bin=3,
    )

    hist = fes["histogram"]
    assert hist.shape == (6, 6)

    metadata = fes["metadata"]
    assert metadata["method"] == "grid"
    assert metadata["min_count_per_bin"] == 3
    assert metadata["smoothed_bins"] > 0
    assert metadata.get("smoothing") == "neighbor_average"

    # Bins neighbouring the populated cell should have received mass.
    zero_fraction = np.count_nonzero(hist == 0.0) / hist.size
    assert zero_fraction < 0.8
