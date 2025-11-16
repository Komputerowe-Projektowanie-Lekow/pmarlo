"""Tests for eigenvector subspace overlap via principal angles."""

from __future__ import annotations

import numpy as np
import pytest

from pmarlo.conformations.kinetic_importance import (
    eigenvector_subspace_overlap_from_bases,
)


def test_identical_1d_subspaces_overlap_is_one() -> None:
    """Same slow direction implies unit overlap."""
    v = np.array([1.0, 2.0, -1.0])
    v = v / np.linalg.norm(v)

    B1 = v.reshape(-1, 1)
    B2 = v.reshape(-1, 1)

    overlap = eigenvector_subspace_overlap_from_bases(B1, B2)
    assert overlap == pytest.approx(1.0, rel=1e-12, abs=1e-12)


def test_orthogonal_1d_subspaces_overlap_is_zero() -> None:
    """Orthogonal slow directions yield zero overlap."""
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])

    B1 = e1.reshape(-1, 1)
    B2 = e2.reshape(-1, 1)

    overlap = eigenvector_subspace_overlap_from_bases(B1, B2)
    assert overlap == pytest.approx(0.0, abs=1e-12)


def test_1d_subspaces_known_angle() -> None:
    """Overlap equals cos^2(theta) for one-dimensional subspaces."""
    theta = np.pi / 3.0  # 60 degrees
    expected = np.cos(theta) ** 2  # 0.25

    v1 = np.array([1.0, 0.0])
    v1 = v1 / np.linalg.norm(v1)

    v2 = np.array([np.cos(theta), np.sin(theta)])
    v2 = v2 / np.linalg.norm(v2)

    B1 = v1.reshape(-1, 1)
    B2 = v2.reshape(-1, 1)

    overlap = eigenvector_subspace_overlap_from_bases(B1, B2)
    assert overlap == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_identical_2d_subspace_invariant_to_basis_rotation() -> None:
    """Subspace rotation must not change the overlap."""
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])

    B1 = np.stack([e1, e2], axis=1)  # shape (3, 2)

    phi = 0.37  # arbitrary rotation angle in that plane
    R = np.array(
        [
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi), np.cos(phi)],
        ]
    )
    B2 = B1 @ R  # same subspace, different basis

    overlap = eigenvector_subspace_overlap_from_bases(B1, B2)
    assert overlap == pytest.approx(1.0, rel=1e-12, abs=1e-12)


def test_partial_overlap_2d_subspaces_plane_intersection() -> None:
    """Planes sharing one direction must produce 0.5 overlap."""
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    e3 = np.array([0.0, 0.0, 1.0])

    B1 = np.stack([e1, e2], axis=1)  # plane 1
    B2 = np.stack([e2, e3], axis=1)  # plane 2

    overlap = eigenvector_subspace_overlap_from_bases(B1, B2)
    assert overlap == pytest.approx(0.5, rel=1e-12, abs=1e-12)


def test_overlap_is_between_zero_and_one_for_random_subspaces() -> None:
    """Random subspaces must always yield overlap inside [0, 1]."""
    rng = np.random.default_rng(123)

    n = 10
    k = 3

    X1 = rng.normal(size=(n, k))
    X2 = rng.normal(size=(n, k))

    B1, _ = np.linalg.qr(X1)
    B2, _ = np.linalg.qr(X2)

    B1 = B1[:, :k]
    B2 = B2[:, :k]

    overlap = eigenvector_subspace_overlap_from_bases(B1, B2)
    assert 0.0 <= overlap <= 1.0
