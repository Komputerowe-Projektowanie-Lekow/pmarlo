import numpy as np
import pytest

from pmarlo.analysis.counting import expected_pairs
from pmarlo.analysis.msm import prepare_msm_discretization


def _make_dataset(train, val=None, test=None):
    splits = {"train": {"X": np.asarray(train, dtype=np.float64)}}
    if val is not None:
        splits["val"] = {"X": np.asarray(val, dtype=np.float64)}
    if test is not None:
        splits["test"] = {"X": np.asarray(test, dtype=np.float64)}
    return {"splits": splits}


def test_prepare_msm_discretization_kmeans_assigns_all_splits():
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(50, 2))
    cluster_b = rng.normal(loc=5.0, scale=0.1, size=(50, 2))
    train = np.vstack([cluster_a, cluster_b])
    rng.shuffle(train)
    val = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.1, size=(5, 2)),
            rng.normal(loc=5.0, scale=0.1, size=(5, 2)),
        ]
    )
    test = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.1, size=(5, 2)),
            rng.normal(loc=5.0, scale=0.1, size=(5, 2)),
        ]
    )
    dataset = _make_dataset(train, val=val, test=test)
    dataset["__shards__"] = [
        {"id": "s0", "start": 0, "stop": int(train.shape[0] / 2)},
        {"id": "s1", "start": int(train.shape[0] / 2), "stop": train.shape[0]},
    ]

    result = prepare_msm_discretization(
        dataset,
        n_microstates=2,
        lag_time=1,
        random_state=0,
    )

    assert set(result.assignments) == {"train", "val", "test"}
    assert 1 <= result.counts.shape[0] <= 2
    assert result.counts.shape[0] == result.counts.shape[1]
    for name, arr in result.assignments.items():
        assert arr.shape[0] == dataset["splits"][name]["X"].shape[0]
    assert result.feature_schema["n_features"] == train.shape[1]
    assert "feature_schema" in result.fingerprint
    assert result.fingerprint["feature_schema"]["n_features"] == train.shape[1]
    assert sum(result.segment_lengths["train"]) == train.shape[0]
    assert all(stride >= 1 for stride in result.segment_strides["train"])
    assert result.counted_pairs["train"] == result.expected_pairs["train"]
    assert result.fingerprint["expected_pairs"] == result.expected_pairs["train"]
    assert result.fingerprint["counted_pairs"] == result.counted_pairs["train"]
    assert "__artifacts__" in dataset
    artifacts = dataset["__artifacts__"]
    assert isinstance(artifacts, dict)
    assert "feature_stats" in artifacts
    assert "state_assignments" in artifacts
    assert "segment_lengths" in artifacts
    assert "expected_pairs" in artifacts
    assert "counted_pairs" in artifacts
    assert "segment_strides" in artifacts
    assert sum(artifacts["segment_lengths"]["train"]) == train.shape[0]
    assert artifacts["expected_pairs"]["train"] == result.expected_pairs["train"]
    assert artifacts["counted_pairs"]["train"] == result.counted_pairs["train"]
    assert all(stride >= 1 for stride in artifacts["segment_strides"]["train"])
    assignment_summary = artifacts["state_assignments"]
    assert assignment_summary["train"]["n_assigned"] == train.shape[0]
    assert assignment_summary["train"]["total"] == train.shape[0]
    masks = result.assignment_masks
    assert set(masks) == {"train", "val", "test"}
    for mask in masks.values():
        assert mask.dtype == np.bool_
        assert mask.shape[0] > 0
        assert mask.all()


def test_prepare_msm_discretization_dbscan_assigns_states():
    train = np.vstack(
        [
            np.random.normal(loc=0.0, scale=0.05, size=(40, 2)),
            np.random.normal(loc=3.0, scale=0.05, size=(40, 2)),
        ]
    )
    dataset = _make_dataset(train)
    dataset["splits"]["train"]["feature_schema"] = {
        "names": ["feature_0", "feature_1"],
        "n_features": 2,
    }

    result = prepare_msm_discretization(
        dataset,
        cluster_mode="dbscan",
        n_microstates=2,
        lag_time=1,
        random_state=0,
        cluster_kwargs={"eps": 0.3, "min_samples": 5},
    )

    assert result.cluster_mode == "dbscan"
    assert result.counts.shape[0] >= 1
    assert result.centers is not None
    assert result.centers.shape[1] == train.shape[1]


def test_prepare_msm_expected_pairs_use_segment_stride_metadata():
    train = np.array([[0.0, 0.0], [0.1, -0.1], [0.2, 0.05], [0.3, -0.2]])
    dataset = _make_dataset(train)
    dataset["__shards__"] = [
        {
            "id": "s0",
            "split": "train",
            "length": train.shape[0],
            "effective_frame_stride": 2,
        }
    ]
    dataset["splits"]["train"]["feature_schema"] = {
        "names": ["feature_0", "feature_1"],
        "n_features": 2,
    }

    result = prepare_msm_discretization(
        dataset,
        n_microstates=2,
        lag_time=1,
        random_state=0,
    )

    assert result.segment_lengths["train"] == [train.shape[0]]
    assert result.segment_strides["train"] == [2]
    expected = expected_pairs([train.shape[0]], 1, [2])
    assert result.expected_pairs["train"] == expected
    assert result.fingerprint["expected_pairs"] == expected


def test_weighted_counts_use_starting_frame_weights():
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
    assert result.segment_lengths["train"] == [train.shape[0]]
    assert result.counted_pairs["train"] == labels.size - 1
    assert result.expected_pairs["train"] == labels.size - 1


def test_grid_mode_discretization_creates_states():
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


def test_prepare_msm_discretization_raises_on_no_assignments(monkeypatch):
    train = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    dataset = _make_dataset(train)

    def _fake_transform(self, X, feature_schema, split_name):
        return np.full(X.shape[0], -1, dtype=np.int32)

    monkeypatch.setattr(
        "pmarlo.analysis.discretize._KMeansDiscretizer.transform",
        _fake_transform,
    )

    from pmarlo.analysis.discretize import NoAssignmentsError

    with pytest.raises(NoAssignmentsError) as excinfo:
        prepare_msm_discretization(
            dataset,
            n_microstates=2,
            lag_time=1,
            random_state=0,
        )

    assert "No state assignments" in str(excinfo.value)
