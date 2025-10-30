from __future__ import annotations

"""Performance benchmarks for distance and dihedral featurization."""

import itertools
import os
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pmarlo.features.builtins import DistancePairFeature
from pmarlo.features.ramachandran import compute_ramachandran, periodic_hist2d

pytestmark = [pytest.mark.perf, pytest.mark.benchmark, pytest.mark.features]

# Optional plugin
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


@pytest.fixture(scope="module")
def reference_traj() -> md.Trajectory:
    """Load a representative protein structure for featurization benchmarks."""

    pdb_path = Path(__file__).resolve().parents[1] / "_assets" / "3gd8-fixed.pdb"
    return md.load(str(pdb_path))


@pytest.fixture
def dense_traj(reference_traj: md.Trajectory) -> md.Trajectory:
    """Expand the reference structure into a multi-frame trajectory."""

    n_frames = 200
    base_xyz = reference_traj.xyz.astype(np.float32, copy=False)
    repeats = (n_frames + reference_traj.n_frames - 1) // reference_traj.n_frames
    tiled = np.tile(base_xyz, (repeats, 1, 1))[:n_frames]

    rng = np.random.default_rng(1234)
    noise = rng.normal(scale=0.02, size=tiled.shape).astype(np.float32)
    xyz = tiled + noise
    return md.Trajectory(xyz, reference_traj.topology)


@pytest.fixture(scope="module")
def calpha_pairs(reference_traj: md.Trajectory) -> np.ndarray:
    """Return sampled C-alpha atom index pairs for distance benchmarks."""

    ca_indices = reference_traj.topology.select("name CA")
    if ca_indices.size < 2:
        pytest.skip("reference structure lacks multiple C-alpha atoms")

    combinations = np.array(list(itertools.combinations(ca_indices, 2)), dtype=np.int32)
    if combinations.shape[0] > 4000:
        rng = np.random.default_rng(42)
        keep = rng.choice(combinations.shape[0], size=4000, replace=False)
        combinations = combinations[keep]
    return combinations


@pytest.fixture(scope="module")
def heavy_atom_pairs(reference_traj: md.Trajectory) -> np.ndarray:
    """Return sampled heavy-atom index pairs for distance featurization."""

    heavy_indices = reference_traj.topology.select("not symbol H")
    if heavy_indices.size < 2:
        pytest.skip("reference structure lacks sufficient heavy atoms")

    combinations = np.array(
        list(itertools.combinations(heavy_indices, 2)), dtype=np.int32
    )
    if combinations.shape[0] > 8000:
        rng = np.random.default_rng(1337)
        keep = rng.choice(combinations.shape[0], size=8000, replace=False)
        combinations = combinations[keep]
    return combinations


@pytest.fixture(scope="module")
def ramachandran_residues(reference_traj: md.Trajectory) -> list[int]:
    """Residue indices with both phi and psi defined."""

    phi, phi_idx = md.compute_phi(reference_traj)
    psi, psi_idx = md.compute_psi(reference_traj)
    if phi.size == 0 or psi.size == 0:
        pytest.skip("reference structure lacks phi/psi definitions")

    top = reference_traj.topology
    phi_res = {top.atom(int(item[1])).residue.index for item in phi_idx}
    psi_res = {top.atom(int(item[2])).residue.index for item in psi_idx}
    residues = sorted(phi_res.intersection(psi_res))
    if not residues:
        pytest.skip("no residues with both phi and psi angles")
    return residues


def test_calpha_distance_matrix(benchmark, dense_traj: md.Trajectory, calpha_pairs):
    """Benchmark pairwise C-alpha distance computation."""

    def _compute():
        return md.compute_distances(dense_traj, calpha_pairs)

    distances = benchmark(_compute)
    assert distances.shape == (dense_traj.n_frames, calpha_pairs.shape[0])
    assert np.all(np.isfinite(distances))


def test_heavy_atom_distance_feature_block(
    benchmark, dense_traj: md.Trajectory, heavy_atom_pairs: np.ndarray
):
    """Benchmark DistancePairFeature across sampled heavy-atom pairs."""

    feature = DistancePairFeature()
    sample_size = min(64, heavy_atom_pairs.shape[0])
    sample_pairs = heavy_atom_pairs[:sample_size]

    def _compute():
        columns = [
            feature.compute(dense_traj, i=int(i), j=int(j)).reshape(-1)
            for i, j in sample_pairs
        ]
        return np.column_stack(columns)

    block = benchmark(_compute)
    assert block.shape == (dense_traj.n_frames, sample_size)
    assert np.all(np.isfinite(block))


def test_ramachandran_single_residue(
    benchmark, dense_traj: md.Trajectory, ramachandran_residues: list[int]
):
    """Benchmark phi/psi computation for a single residue."""

    target = ramachandran_residues[len(ramachandran_residues) // 2]

    def _compute():
        return compute_ramachandran(dense_traj, selection=target)

    angles = benchmark(_compute)
    assert angles.shape == (dense_traj.n_frames, 2)
    assert np.all(np.isfinite(angles))


def test_ramachandran_batch_processing(
    benchmark, dense_traj: md.Trajectory, ramachandran_residues: list[int]
):
    """Benchmark repeated Ramachandran calculations across multiple residues."""

    subset = ramachandran_residues[: min(10, len(ramachandran_residues))]

    def _compute():
        return [compute_ramachandran(dense_traj, selection=res) for res in subset]

    results = benchmark(_compute)
    assert len(results) == len(subset)
    for angles in results:
        assert angles.shape == (dense_traj.n_frames, 2)
        assert np.all(np.isfinite(angles))


def test_ramachandran_histogram_generation(
    benchmark, dense_traj: md.Trajectory, ramachandran_residues: list[int]
):
    """Benchmark periodic histogram construction from dihedral angles."""

    target = ramachandran_residues[0]
    angles = compute_ramachandran(dense_traj, selection=target)

    def _compute():
        return periodic_hist2d(angles[:, 0], angles[:, 1], bins=(72, 72))

    H, x_edges, y_edges = benchmark(_compute)
    assert H.shape == (72, 72)
    assert x_edges.shape[0] == 72
    assert y_edges.shape[0] == 72
    assert np.all(H >= 0.0)
