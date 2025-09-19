import importlib
import pathlib
import sys
import types

import numpy as np


def _prepare_imports():
    if "pmarlo.analysis.msm" in sys.modules:
        del sys.modules["pmarlo.analysis.msm"]
    if "pmarlo.analysis" in sys.modules:
        del sys.modules["pmarlo.analysis"]
    if "pmarlo" in sys.modules:
        del sys.modules["pmarlo"]

    base = pathlib.Path("src/pmarlo")
    pmarlo_pkg = types.ModuleType("pmarlo")
    pmarlo_pkg.__path__ = [str(base)]
    sys.modules["pmarlo"] = pmarlo_pkg

    ml_pkg = sys.modules.setdefault("pmarlo.ml", types.ModuleType("pmarlo.ml"))
    if not hasattr(ml_pkg, "__path__"):
        ml_pkg.__path__ = []  # pragma: no cover - defensive for namespace packages

    deeptica_pkg = types.ModuleType("pmarlo.ml.deeptica")
    deeptica_pkg.__path__ = []
    sys.modules["pmarlo.ml.deeptica"] = deeptica_pkg

    whitening_mod = types.ModuleType("pmarlo.ml.deeptica.whitening")

    def _identity_transform(Y, mean, W, already_applied):
        return np.asarray(Y, dtype=np.float64)

    whitening_mod.apply_output_transform = _identity_transform
    sys.modules["pmarlo.ml.deeptica.whitening"] = whitening_mod
    deeptica_pkg.apply_output_transform = _identity_transform

    analysis_mod = importlib.import_module("pmarlo.analysis.msm")
    return analysis_mod.prepare_msm_discretization


def _make_dataset(train, val=None, test=None):
    splits = {"train": {"X": np.asarray(train, dtype=np.float64)}}
    if val is not None:
        splits["val"] = {"X": np.asarray(val, dtype=np.float64)}
    if test is not None:
        splits["test"] = {"X": np.asarray(test, dtype=np.float64)}
    return {"splits": splits}


def test_prepare_msm_discretization_kmeans_assigns_all_splits():
    prepare_msm_discretization = _prepare_imports()
    train = np.array([[0.0, 0.0], [0.1, -0.1], [4.0, 4.0], [4.2, 3.9]])
    val = np.array([[0.05, 0.05], [4.1, 4.1]])
    test = np.array([[0.2, -0.05]])
    dataset = _make_dataset(train, val=val, test=test)

    result = prepare_msm_discretization(
        dataset,
        n_microstates=2,
        lag_time=1,
        random_state=0,
    )

    assert set(result.assignments) == {"train", "val", "test"}
    assert result.counts.shape == (2, 2)
    for name, arr in result.assignments.items():
        assert arr.shape[0] == dataset["splits"][name]["X"].shape[0]


def test_weighted_counts_use_starting_frame_weights():
    prepare_msm_discretization = _prepare_imports()
    train = np.array(
        [
            [0.0, 0.0],
            [0.05, -0.05],
            [5.0, 5.0],
            [5.1, 5.2],
        ],
        dtype=np.float64,
    )
    weights = np.array([1.0, 0.5, 2.0, 3.0])
    dataset = _make_dataset(train)

    result = prepare_msm_discretization(
        dataset,
        n_microstates=2,
        lag_time=1,
        frame_weights={"train": weights},
        random_state=0,
    )

    labels = result.assignments["train"]
    expected = np.zeros_like(result.counts)
    for idx in range(labels.size - 1):
        expected[labels[idx], labels[idx + 1]] += weights[idx]
    assert np.allclose(result.counts, expected)


def test_grid_mode_discretization_creates_states():
    prepare_msm_discretization = _prepare_imports()
    train = np.array([[0.0], [0.2], [1.0], [1.2]])
    dataset = _make_dataset(train)

    result = prepare_msm_discretization(
        dataset,
        cluster_mode="grid",
        n_microstates=4,
        lag_time=1,
    )

    assert result.cluster_mode == "grid"
    assert result.counts.shape[0] >= 1
    assert result.transition_matrix.shape == result.counts.shape


def test_frame_weights_length_mismatch_raises():
    prepare_msm_discretization = _prepare_imports()
    train = np.array([[0.0, 0.0], [1.0, 1.0]])
    dataset = _make_dataset(train)

    try:
        prepare_msm_discretization(
            dataset,
            n_microstates=2,
            frame_weights={"train": np.array([1.0])},
        )
    except ValueError as exc:
        assert "Frame weights" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched weights")
