import numpy as np
import pytest

from pmarlo.analysis.msm import prepare_msm_discretization


def _make_dataset(train, val=None, test=None):
    splits = {"train": {"X": np.asarray(train, dtype=np.float64)}}
    if val is not None:
        splits["val"] = {"X": np.asarray(val, dtype=np.float64)}
    if test is not None:
        splits["test"] = {"X": np.asarray(test, dtype=np.float64)}
    return {"splits": splits}


def test_prepare_msm_discretization_kmeans_assigns_all_splits():
    train = np.array([[0.0, 0.0], [0.1, -0.1], [4.0, 4.0], [4.2, 3.9]])
    val = np.array([[0.05, 0.05], [4.1, 4.1]])
    test = np.array([[0.2, -0.05], [4.05, 4.02]])
    dataset = _make_dataset(train, val=val, test=test)
    dataset["__shards__"] = [
        {"id": "s0", "start": 0, "stop": 2},
        {"id": "s1", "start": 2, "stop": 3},
        {"id": "s2", "start": 3, "stop": 4},
    ]

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
    assert result.feature_schema["n_features"] == train.shape[1]
    assert "feature_schema" in result.fingerprint
    assert result.fingerprint["feature_schema"]["n_features"] == train.shape[1]
    assert result.segment_lengths["train"] == [2, 1, 1]
    assert result.segment_strides["train"] == [1, 1, 1]
    assert result.counted_pairs["train"] == result.expected_pairs["train"] == 1
    assert result.fingerprint["expected_pairs"] == 1
    assert result.fingerprint["counted_pairs"] == 1
    assert "__artifacts__" in dataset
    artifacts = dataset["__artifacts__"]
    assert isinstance(artifacts, dict)
    assert "feature_stats" in artifacts
    assert "state_assignments" in artifacts
    assert "segment_lengths" in artifacts
    assert "expected_pairs" in artifacts
    assert "counted_pairs" in artifacts
    assert "segment_strides" in artifacts
    assert artifacts["segment_lengths"]["train"] == [2, 1, 1]
    assert artifacts["expected_pairs"]["train"] == 1
    assert artifacts["counted_pairs"]["train"] == 1
    assert artifacts["segment_strides"]["train"] == [1, 1, 1]
    assignment_summary = artifacts["state_assignments"]
    assert assignment_summary["train"]["n_assigned"] == train.shape[0]
    assert assignment_summary["train"]["total"] == train.shape[0]
    masks = result.assignment_masks
    assert set(masks) == {"train", "val", "test"}
    for mask in masks.values():
        assert mask.dtype == np.bool_
        assert mask.shape[0] > 0
        assert mask.all()


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
