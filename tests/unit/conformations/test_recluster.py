"""Contract tests for the `_recluster` helper."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pytest

pytest.importorskip("sklearn")

from pmarlo.conformations.kinetic_importance import KineticImportanceScore


class DummyClusterer:
    """Minimal wrapper that exposes the real `_recluster` implementation."""

    def __init__(self, traj_lengths: Sequence[int]):
        if not isinstance(traj_lengths, Iterable):
            raise TypeError("traj_lengths must be an iterable of integers")
        self._traj_lengths = list(traj_lengths)


DummyClusterer._recluster = KineticImportanceScore._recluster


def test_recluster_returns_one_trajectory_per_input_trajectory() -> None:
    """Two well-separated trajectories should produce distinct label sequences."""

    # 1st trajectory: 4 frames near the origin
    traj1 = np.zeros((4, 2), dtype=float)

    # 2nd trajectory: 3 frames near (10, 0)
    traj2 = np.ones((3, 2), dtype=float) * 10.0

    features = np.vstack([traj1, traj2])
    traj_lengths = [len(traj1), len(traj2)]

    clusterer = DummyClusterer(traj_lengths=traj_lengths)
    discrete_trajs = clusterer._recluster(features, n_clusters=2)

    # One output per input trajectory and matching lengths
    assert len(discrete_trajs) == 2
    assert discrete_trajs[0].shape == (4,)
    assert discrete_trajs[1].shape == (3,)

    # Consistent labels within a trajectory
    labels_traj1 = set(discrete_trajs[0].tolist())
    labels_traj2 = set(discrete_trajs[1].tolist())
    assert len(labels_traj1) == 1
    assert len(labels_traj2) == 1

    # Different clusters across trajectories
    assert labels_traj1.isdisjoint(labels_traj2)


def test_recluster_single_cluster_gives_single_label_everywhere() -> None:
    """Requesting a single cluster forces one shared label."""

    traj1 = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=float)
    traj2 = np.array([[5.0, 5.0], [5.1, 5.2], [4.9, 5.1]], dtype=float)
    traj3 = np.array([[1.0, -1.0]], dtype=float)

    features = np.vstack([traj1, traj2, traj3])
    traj_lengths = [len(traj1), len(traj2), len(traj3)]

    clusterer = DummyClusterer(traj_lengths=traj_lengths)
    discrete_trajs = clusterer._recluster(features, n_clusters=1)

    flat_labels = np.concatenate(discrete_trajs)
    unique_labels = np.unique(flat_labels)

    assert unique_labels.shape == (1,)


def test_recluster_raises_if_lengths_do_not_match_features() -> None:
    """The helper should fail loudly when trajectory lengths mismatch."""

    features = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )

    traj_lengths = [3, 3]  # Sum does not equal the number of labels (5)

    clusterer = DummyClusterer(traj_lengths=traj_lengths)

    with pytest.raises(ValueError):
        clusterer._recluster(features, n_clusters=2)
