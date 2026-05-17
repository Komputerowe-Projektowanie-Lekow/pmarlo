"""High-level feature extraction from mdtraj trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mdtraj

__all__ = ["featurize_trajectory"]

_SUPPORTED = ("phi_psi", "ca_distances", "backbone_torsions")


def featurize_trajectory(
    traj: "mdtraj.Trajectory",
    feature_type: str = "phi_psi",
) -> np.ndarray:
    """Extract a feature matrix from an mdtraj Trajectory.

    Parameters
    ----------
    traj:
        Input trajectory.
    feature_type:
        One of:

        * ``"phi_psi"`` – backbone dihedral angles φ and ψ (radians).
          Good default for small peptides such as alanine dipeptide.
        * ``"ca_distances"`` – pairwise Cα distances (nm).
        * ``"backbone_torsions"`` – φ, ψ, and χ₁ where available.

    Returns
    -------
    np.ndarray of shape ``(n_frames, n_features)``
    """
    import mdtraj

    if feature_type == "phi_psi":
        _, phi = mdtraj.compute_phi(traj)
        _, psi = mdtraj.compute_psi(traj)
        return np.concatenate([phi, psi], axis=1)

    if feature_type == "ca_distances":
        ca = traj.topology.select("name CA")
        if len(ca) < 2:
            raise ValueError("Topology has fewer than 2 Cα atoms.")
        pairs = np.array(
            [(ca[i], ca[j]) for i in range(len(ca)) for j in range(i + 1, len(ca))]
        )
        return mdtraj.compute_distances(traj, pairs)

    if feature_type == "backbone_torsions":
        _, phi = mdtraj.compute_phi(traj)
        _, psi = mdtraj.compute_psi(traj)
        _, chi1 = mdtraj.compute_chi1(traj)
        arrays = [phi, psi]
        if chi1.shape[1] > 0:
            arrays.append(chi1)
        return np.concatenate(arrays, axis=1)

    raise ValueError(
        f"Unknown feature_type {feature_type!r}. Choose one of {_SUPPORTED}."
    )
