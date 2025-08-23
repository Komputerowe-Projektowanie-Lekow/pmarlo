from __future__ import annotations

from typing import Tuple, cast

import mdtraj as md  # type: ignore
import numpy as np

from .base import register_feature


def _wrap_to_minus_pi_pi(angles: np.ndarray) -> np.ndarray:
    return ((angles + np.pi) % (2 * np.pi)) - np.pi


def _compute_phi(traj_in: md.Trajectory) -> Tuple[np.ndarray, np.ndarray]:
    angles, idx = md.compute_phi(traj_in)
    return _wrap_to_minus_pi_pi(angles), idx


def _compute_psi(traj_in: md.Trajectory) -> Tuple[np.ndarray, np.ndarray]:
    angles, idx = md.compute_psi(traj_in)
    return _wrap_to_minus_pi_pi(angles), idx


def _labels_from_indices(
    traj_in: md.Trajectory, indices: np.ndarray, kind: str, atom_pos: int
) -> list[str]:
    labels_local: list[str] = []
    try:
        top = traj_in.topology
        for four in indices:
            atom_index = int(four[atom_pos])
            resid = int(top.atom(atom_index).residue.index)
            labels_local.append(f"{kind}:res{resid}")
    except Exception:
        labels_local = [f"{kind}_{i}" for i in range(indices.shape[0])]
    return labels_local


class PhiPsiFeature:
    name = "phi_psi"

    def __init__(self) -> None:
        # periodic per returned column: [phi..., psi...]
        self._periodic: np.ndarray | None = None
        # last computed column labels (residue-aware when possible)
        self.labels: list[str] | None = None

    def compute(self, traj: md.Trajectory, **kwargs) -> np.ndarray:
        phi, phi_idx = _compute_phi(traj)
        psi, psi_idx = _compute_psi(traj)

        if phi.size == 0 and psi.size == 0:
            self.labels = []
            return np.zeros((traj.n_frames, 0), dtype=float)

        columns: list[np.ndarray] = []
        labels: list[str] = []
        if phi.size:
            columns.append(phi)
            labels.extend(_labels_from_indices(traj, phi_idx, "phi", 1))
        if psi.size:
            columns.append(psi)
            labels.extend(_labels_from_indices(traj, psi_idx, "psi", 2))

        X = np.hstack(columns) if columns else np.zeros((traj.n_frames, 0), dtype=float)

        # periodic flags per column
        n_cols = X.shape[1]
        self._periodic = np.ones((n_cols,), dtype=bool)

        # align labels length if possible
        self.labels = labels if len(labels) == n_cols else None
        return X

    def is_periodic(self) -> np.ndarray:
        if self._periodic is None:
            # default unknown -> False length 0; caller should compute first
            return np.zeros((0,), dtype=bool)
        return self._periodic


# Register built-ins at import
register_feature(PhiPsiFeature())


class RadiusOfGyrationFeature:
    name = "Rg"

    def __init__(self) -> None:
        self._periodic = np.array([False], dtype=bool)
        self.labels: list[str] | None = None

    def compute(self, traj: md.Trajectory, **kwargs) -> np.ndarray:
        rg = md.compute_rg(traj)
        # expose label for consistency
        self.labels = ["Rg"]
        return cast(np.ndarray, rg.reshape(-1, 1).astype(float))

    def is_periodic(self) -> np.ndarray:
        return self._periodic


register_feature(RadiusOfGyrationFeature())


class DistancePairFeature:
    name = "distance_pair"

    def __init__(self) -> None:
        self._periodic = np.array([False], dtype=bool)
        self.labels: list[str] | None = None

    def compute(self, traj: md.Trajectory, **kwargs) -> np.ndarray:
        i = int(kwargs.get("i", -1))
        j = int(kwargs.get("j", -1))
        n_atoms = traj.n_atoms
        if not (0 <= i < n_atoms) or not (0 <= j < n_atoms):
            raise ValueError("Atom indices out of range")

        pairs = [[i, j]]
        d = md.compute_distances(traj, pairs)
        # Replace possible NaN/inf values produced by mdtraj with zeros to
        # keep the feature array numerically stable.
        np.nan_to_num(d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self.labels = [f"dist:atoms:{i}-{j}"]
        return cast(np.ndarray, d.astype(float, copy=False))

    def is_periodic(self) -> np.ndarray:
        return self._periodic


register_feature(DistancePairFeature())


class Chi1Feature:
    name = "chi1"

    def __init__(self) -> None:
        self._periodic: np.ndarray | None = None
        self.labels: list[str] | None = None

    def compute(self, traj: md.Trajectory, **kwargs) -> np.ndarray:
        try:
            chi1_angles, chi1_idx = md.compute_chi1(traj)
        except Exception:
            # mdtraj may not support chi1 on all topologies
            self.labels = []
            return np.zeros((traj.n_frames, 0), dtype=float)
        # wrap
        chi1 = ((chi1_angles + np.pi) % (2 * np.pi)) - np.pi
        if chi1.size == 0:
            self.labels = []
            return np.zeros((traj.n_frames, 0), dtype=float)
        self._periodic = np.ones((chi1.shape[1],), dtype=bool)
        # residue-aware labels if possible (use second index like phi)
        labels: list[str] = []
        try:
            top = traj.topology
            for four in chi1_idx:
                atom_index = int(four[1])
                resid = int(top.atom(atom_index).residue.index)
                labels.append(f"chi1:res{resid}")
        except Exception:
            labels = [f"chi1_{i}" for i in range(chi1.shape[1])]
        self.labels = labels
        return cast(np.ndarray, chi1)

    def is_periodic(self) -> np.ndarray:
        if self._periodic is None:
            return np.zeros((0,), dtype=bool)
        return self._periodic


register_feature(Chi1Feature())


class SASAFeature:
    name = "sasa"

    def __init__(self) -> None:
        self._periodic = np.array([False], dtype=bool)

    def compute(self, traj: md.Trajectory, **kwargs) -> np.ndarray:
        try:
            import mdtraj as _md

            sasa = _md.shrake_rupley(traj, mode="residue")  # (n_frames, n_residues)
            # Sum per frame for compact scalar feature
            return cast(np.ndarray, np.sum(sasa, axis=1, keepdims=True).astype(float))
        except Exception:
            return np.zeros((traj.n_frames, 1), dtype=float)

    def is_periodic(self) -> np.ndarray:
        return self._periodic


register_feature(SASAFeature())


class HBondsCountFeature:
    name = "hbonds_count"

    def __init__(self) -> None:
        self._periodic = np.array([False], dtype=bool)

    def compute(self, traj: md.Trajectory, **kwargs) -> np.ndarray:
        try:
            import mdtraj as _md

            hbonds = _md.baker_hubbard(traj, periodic=True)
            # Count H-bonds per frame by checking distances/angles (rough proxy: reuse pairs across frames)
            # mdtraj does not provide per-frame counts directly; fall back to static count
            count = np.full((traj.n_frames, 1), float(len(hbonds)), dtype=float)
            return count
        except Exception:
            return np.zeros((traj.n_frames, 1), dtype=float)

    def is_periodic(self) -> np.ndarray:
        return self._periodic


register_feature(HBondsCountFeature())


class SecondaryStructureFractionFeature:
    name = "ssfrac"

    def __init__(self) -> None:
        self._periodic = np.array([False, False, False], dtype=bool)

    def compute(self, traj: md.Trajectory, **kwargs) -> np.ndarray:
        try:
            import mdtraj as _md

            dssp = _md.compute_dssp(traj)  # (n_frames, n_residues) with letters
            # Fractions: helix (H,G,I), sheet (E,B), coil (others)
            helix_set = {"H", "G", "I"}
            sheet_set = {"E", "B"}
            out = np.zeros((traj.n_frames, 3), dtype=float)
            for f in range(traj.n_frames):
                row = dssp[f]
                n = float(len(row)) if len(row) > 0 else 1.0
                helix = sum(1 for c in row if c in helix_set) / n
                sheet = sum(1 for c in row if c in sheet_set) / n
                coil = 1.0 - helix - sheet
                out[f, :] = (helix, sheet, max(0.0, coil))
            return out
        except Exception:
            return np.zeros((traj.n_frames, 3), dtype=float)

    def is_periodic(self) -> np.ndarray:
        return self._periodic


register_feature(SecondaryStructureFractionFeature())


class ContactsPairFeature:
    name = "contacts_pair"

    def __init__(self) -> None:
        self._periodic = np.array([False], dtype=bool)

    def compute(self, traj: md.Trajectory, **kwargs) -> np.ndarray:
        i = int(kwargs.get("i", -1))
        j = int(kwargs.get("j", -1))
        rcut = float(kwargs.get("rcut", 0.5))
        n_atoms = traj.n_atoms
        if rcut <= 0:
            raise ValueError("rcut must be positive")
        if not (0 <= i < n_atoms) or not (0 <= j < n_atoms):
            raise ValueError("Atom indices out of range")
        pairs = [[i, j]]
        d = md.compute_distances(traj, pairs)
        # NaNs in distances imply missing coordinates; treat them as infinite so
        # that the contact is reported as absent.
        np.nan_to_num(d, copy=False, nan=np.inf, posinf=np.inf, neginf=np.inf)
        return cast(np.ndarray, (d <= rcut).astype(float, copy=False))

    def is_periodic(self) -> np.ndarray:
        return self._periodic


register_feature(ContactsPairFeature())
