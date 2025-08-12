# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Enhanced Markov State Model analysis with TRAM/dTRAM and comprehensive reporting.

This module provides advanced MSM analysis capabilities including:
- TRAM/dTRAM for multi-temperature data
- Free energy surface generation
- State table export
- Implied timescales analysis
- Representative structure extraction
- Comprehensive visualization
"""

import json
import logging
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from scipy import constants
from scipy.ndimage import gaussian_filter
from scipy.sparse import csc_matrix, issparse, save_npz
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


class EnhancedMSM:
    """
    Enhanced Markov State Model with advanced analysis and reporting capabilities.

    This class provides comprehensive MSM analysis including multi-temperature
    data handling, free energy surface generation, and detailed reporting.
    """

    def __init__(
        self,
        trajectory_files: Optional[Union[str, List[str]]] = None,
        topology_file: Optional[str] = None,
        temperatures: Optional[List[float]] = None,
        output_dir: str = "output/msm_analysis",
    ):
        """
        Initialize the Enhanced MSM analyzer.

        Args:
            trajectory_files: Single trajectory file or list of files
            topology_file: Topology file (PDB) for the system
            temperatures: List of temperatures for TRAM analysis
            output_dir: Directory for output files
        """
        self.trajectory_files = (
            trajectory_files
            if isinstance(trajectory_files, list)
            else [trajectory_files] if trajectory_files else []
        )
        self.topology_file = topology_file
        self.temperatures = temperatures or [300.0]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Analysis data - Fixed: Added proper type annotations
        self.trajectories: List[md.Trajectory] = []  # Fix: Added type annotation
        self.dtrajs: List[np.ndarray] = (
            []
        )  # Fix: Added type annotation (discrete trajectories)
        self.features: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.n_states = 0

        # MSM data - Fixed: Initialize with proper types instead of None
        self.transition_matrix: Optional[np.ndarray] = (
            None  # Fix: Will be properly initialized later
        )
        self.count_matrix: Optional[np.ndarray] = (
            None  # Fix: Will be properly initialized later
        )
        self.stationary_distribution: Optional[np.ndarray] = None
        self.free_energies: Optional[np.ndarray] = None
        self.lag_time = 20  # Default lag time
        self.frame_stride: Optional[int] = None  # Optional: frames per saved sample

        # TRAM data
        self.tram_weights: Optional[np.ndarray] = None
        self.multi_temp_counts: Dict[float, Dict[Tuple[int, int], float]] = (
            {}
        )  # Fix: Added proper type annotation

        # Analysis results - Fixed: Initialize with proper type annotation
        self.implied_timescales: Optional[Dict[str, Any]] = None
        self.state_table: Optional[pd.DataFrame] = (
            None  # Fix: Will be DataFrame when created
        )
        self.fes_data: Optional[Dict[str, Any]] = None

        logger.info(
            f"Enhanced MSM initialized for {len(self.trajectory_files)} trajectories"
        )

    def load_trajectories(self, stride: int = 1):
        """
        Load trajectory data for analysis.

        Args:
            stride: Stride for loading frames (1 = every frame)
        """
        logger.info("Loading trajectory data...")

        self.trajectories = []
        for i, traj_file in enumerate(self.trajectory_files):
            if Path(traj_file).exists():
                traj = md.load(traj_file, top=self.topology_file, stride=stride)
                self.trajectories.append(traj)
                logger.info(f"Loaded trajectory {i+1}: {traj.n_frames} frames")
            else:
                logger.warning(f"Trajectory file not found: {traj_file}")

        if not self.trajectories:
            raise ValueError("No trajectories loaded successfully")

        logger.info(f"Total trajectories loaded: {len(self.trajectories)}")
        # Track total frames for adaptive clustering heuristics
        try:
            self.total_frames: Optional[int] = int(
                sum(int(t.n_frames) for t in self.trajectories)
            )
        except Exception:
            self.total_frames = None

    def save_phi_psi_scatter_diagnostics(
        self,
        *,
        max_residues: int = 6,
        exclude_special: bool = True,
        sample_per_residue: int = 2000,
        filename: str = "diagnostics_phi_psi_scatter.png",
    ) -> Optional[Path]:
        """Save a raw φ/ψ scatter plot across several residues for sanity check.

        - Uses multiple internal residues that have both φ and ψ defined
        - Optionally excludes termini and Gly/Pro (distinctive Ramachandran regions)
        - Wraps to degrees in [-180, 180]
        """
        try:
            if not self.trajectories:
                return None

            all_phi, all_psi, selected_meta = self._init_phi_psi_buffers()

            for traj in self.trajectories:
                dihedrals = self._compute_traj_dihedrals(traj)
                if dihedrals is None:
                    continue
                phi_indices, phi, psi_indices, psi = dihedrals

                phi_map, psi_map, common_res = self._map_residues_to_dihedrals(
                    traj, phi_indices, psi_indices
                )

                candidates = self._build_phi_psi_candidates(
                    traj, common_res, phi_map, psi_map, exclude_special
                )
                if not candidates:
                    continue

                selected = self._select_candidate_residues(candidates, max_residues)
                phi_wrap, psi_wrap = self._wrap_dihedral_angles_in_degrees(phi, psi)

                self._accumulate_samples_for_selected(
                    selected,
                    phi_wrap,
                    psi_wrap,
                    sample_per_residue,
                    all_phi,
                    all_psi,
                    selected_meta,
                )

            if not all_phi or not all_psi:
                logger.warning("φ/ψ diagnostics: no data collected; skipping plot")
                return None

            out_path = self._plot_phi_psi_scatter_and_save(all_phi, all_psi, filename)
            self._save_phi_psi_metadata(selected_meta, max_residues)
            logger.info(f"Saved φ/ψ diagnostic scatter: {out_path}")
            return out_path
        except Exception as e:
            logger.warning(f"φ/ψ diagnostics failed: {e}")
            return None

    # ---- φ/ψ diagnostics helpers (split to address C901) ----

    def _init_phi_psi_buffers(self) -> tuple[list[float], list[float], list[dict]]:
        all_phi: list[float] = []
        all_psi: list[float] = []
        selected_meta: list[dict] = []
        return all_phi, all_psi, selected_meta

    def _compute_traj_dihedrals(
        self, traj: md.Trajectory
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        phi_indices, phi = md.compute_phi(traj)
        psi_indices, psi = md.compute_psi(traj)
        if phi.size == 0 or psi.size == 0:
            return None
        return phi_indices, phi, psi_indices, psi

    def _map_residues_to_dihedrals(
        self,
        traj: md.Trajectory,
        phi_indices: np.ndarray,
        psi_indices: np.ndarray,
    ) -> tuple[Dict[int, int], Dict[int, int], List[int]]:
        # Map dihedral columns to residue indices via second atom (N_i)
        phi_res_ids = [
            int(traj.topology.atom(int(idx[1])).residue.index) for idx in phi_indices
        ]
        psi_res_ids = [
            int(traj.topology.atom(int(idx[1])).residue.index) for idx in psi_indices
        ]
        phi_map = {rid: col for col, rid in enumerate(phi_res_ids)}
        psi_map = {rid: col for col, rid in enumerate(psi_res_ids)}
        common_res = sorted(set(phi_map).intersection(psi_map))
        return phi_map, psi_map, common_res

    def _build_phi_psi_candidates(
        self,
        traj: md.Trajectory,
        common_res: List[int],
        phi_map: Dict[int, int],
        psi_map: Dict[int, int],
        exclude_special: bool,
    ) -> list[tuple[int, int, int, str]]:
        candidates: list[tuple[int, int, int, str]] = []
        n_residues = traj.topology.n_residues
        for rid in common_res:
            # Exclude termini (lack one of the dihedrals typically)
            if rid <= 0 or rid >= n_residues - 1:
                continue
            res = traj.topology.residue(rid)
            res_name = str(res.name).upper()
            if exclude_special and res_name in {"GLY", "PRO"}:
                continue
            candidates.append((rid, phi_map[rid], psi_map[rid], res_name))
        return candidates

    def _select_candidate_residues(
        self, candidates: list[tuple[int, int, int, str]], max_residues: int
    ) -> list[tuple[int, int, int, str]]:
        step = max(1, len(candidates) // max_residues)
        return [candidates[i] for i in range(0, len(candidates), step)][:max_residues]

    def _wrap_dihedral_angles_in_degrees(
        self, phi: np.ndarray, psi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi_deg = np.degrees(phi)
        psi_deg = np.degrees(psi)
        phi_wrap = ((phi_deg + 180.0) % 360.0) - 180.0
        psi_wrap = ((psi_deg + 180.0) % 360.0) - 180.0
        return phi_wrap, psi_wrap

    def _accumulate_samples_for_selected(
        self,
        selected: list[tuple[int, int, int, str]],
        phi_wrap: np.ndarray,
        psi_wrap: np.ndarray,
        sample_per_residue: int,
        all_phi: list[float],
        all_psi: list[float],
        selected_meta: list[dict],
    ) -> None:
        for rid, cphi, cpsi, res_name in selected:
            x = phi_wrap[:, cphi].astype(float)
            y = psi_wrap[:, cpsi].astype(float)
            # Subsample evenly to avoid massive plots
            if x.size > sample_per_residue:
                stride = max(1, x.size // sample_per_residue)
                x = x[::stride]
                y = y[::stride]
            all_phi.extend(x.tolist())
            all_psi.extend(y.tolist())
            selected_meta.append({"residue_index": rid, "name": res_name})

    def _plot_phi_psi_scatter_and_save(
        self, all_phi: list[float], all_psi: list[float], filename: str
    ) -> Path:
        plt.figure(figsize=(7, 6))
        plt.scatter(all_phi, all_psi, s=4, alpha=0.25, c="tab:blue")
        plt.xlim([-180, 180])
        plt.ylim([-180, 180])
        plt.xlabel("ϕ (deg)")
        plt.ylabel("ψ (deg)")
        plt.title("Raw ϕ/ψ scatter (multi-residue)")
        out_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        return out_path

    def _save_phi_psi_metadata(
        self, selected_meta: list[dict], max_residues: int
    ) -> None:
        try:
            meta_path = self.output_dir / "diagnostics_phi_psi_meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"selected_residues": selected_meta[:max_residues]}, f, indent=2
                )
        except Exception:
            # Best-effort metadata save; do not raise
            pass

    def compute_features(
        self, feature_type: str = "phi_psi", n_features: Optional[int] = None
    ):
        """
        Compute features from trajectory data.

        Args:
            feature_type: Type of features to compute
                ('phi_psi', 'distances', 'contacts')
            n_features: Number of features to compute (auto if None)
        """
        self._log_compute_features_start(feature_type)
        all_features: List[np.ndarray] = []
        for traj in self.trajectories:
            traj_features = self._compute_features_for_traj(
                traj, feature_type, n_features
            )
            all_features.append(traj_features)
        self.features = self._combine_all_features(all_features)
        self._log_features_shape()
        # Optional: apply TICA projection (2-5 components) when requested
        self._maybe_apply_tica(feature_type, n_features)

    # ---------------- Feature computation helper methods ----------------

    def _log_compute_features_start(self, feature_type: str) -> None:
        logger.info(f"Computing {feature_type} features...")

    def _compute_features_for_traj(
        self, traj: md.Trajectory, feature_type: str, n_features: Optional[int]
    ) -> np.ndarray:
        ft = feature_type.lower()
        if ft.startswith("phi_psi_distances"):
            return self._compute_phi_psi_plus_distance_features(traj, n_features)
        if ft.startswith("phi_psi"):
            return self._compute_phi_psi_features(traj)
        if ft == "distances":
            return self._compute_distance_features(traj, n_features)
        if ft == "contacts":
            return self._compute_contact_features(traj)
        raise ValueError(f"Unknown feature type: {feature_type}")

    def _compute_phi_psi_features(self, traj: md.Trajectory) -> np.ndarray:
        phi_angles, _ = md.compute_phi(traj)
        psi_angles, _ = md.compute_psi(traj)
        features: List[np.ndarray] = []
        self._maybe_extend_with_trig(features, phi_angles)
        self._maybe_extend_with_trig(features, psi_angles)
        if features:
            return np.hstack(features)
        logger.warning("No dihedral angles found, using Cartesian coordinates")
        return self._fallback_cartesian_features(traj)

    def _maybe_extend_with_trig(
        self, features: List[np.ndarray], angles: np.ndarray
    ) -> None:
        if angles.shape[1] > 0:
            features.extend([np.cos(angles), np.sin(angles)])

    def _fallback_cartesian_features(self, traj: md.Trajectory) -> np.ndarray:
        return traj.xyz.reshape(traj.n_frames, -1)

    def _compute_phi_psi_plus_distance_features(
        self, traj: md.Trajectory, n_distance_features: Optional[int]
    ) -> np.ndarray:
        """Concatenate φ/ψ trig-expanded features with selected Cα distances."""
        phi_psi = self._compute_phi_psi_features(traj)
        dists = self._compute_distance_features(traj, n_distance_features)
        if phi_psi.shape[0] != dists.shape[0]:
            min_len = min(phi_psi.shape[0], dists.shape[0])
            phi_psi = phi_psi[:min_len]
            dists = dists[:min_len]
        return np.hstack([phi_psi, dists])

    def _compute_distance_features(
        self, traj: md.Trajectory, n_features: Optional[int]
    ) -> np.ndarray:
        ca_indices = traj.topology.select("name CA")
        self._validate_distance_atoms(ca_indices)
        n_pairs = self._determine_num_pairs(ca_indices, n_features)
        pairs = self._select_distance_pairs(ca_indices, n_pairs)
        return md.compute_distances(traj, pairs)

    def _validate_distance_atoms(self, ca_indices: np.ndarray) -> None:
        if len(ca_indices) < 2:
            raise ValueError("Insufficient Cα atoms for distance features")

    def _determine_num_pairs(
        self, ca_indices: np.ndarray, n_features: Optional[int]
    ) -> int:
        total_pairs = len(ca_indices) * (len(ca_indices) - 1) // 2
        if n_features:
            return min(n_features, total_pairs)
        return min(200, total_pairs)

    def _select_distance_pairs(
        self, ca_indices: np.ndarray, n_pairs: int
    ) -> List[List[int]]:
        pairs: List[List[int]] = []
        for i in range(0, len(ca_indices), 3):
            for j in range(i + 3, len(ca_indices), 3):
                pairs.append([int(ca_indices[i]), int(ca_indices[j])])
                if len(pairs) >= n_pairs:
                    break
            if len(pairs) >= n_pairs:
                break
        return pairs

    def _compute_contact_features(self, traj: md.Trajectory) -> np.ndarray:
        contacts, _pairs = md.compute_contacts(traj, contacts="all", scheme="ca")
        return contacts

    def _combine_all_features(self, feature_blocks: List[np.ndarray]) -> np.ndarray:
        return np.vstack(feature_blocks)

    def _log_features_shape(self) -> None:
        if self.features is not None:
            logger.info(f"Features computed: {self.features.shape}")

    def cluster_features(self, n_clusters: int = 100, algorithm: str = "kmeans"):
        """
        Cluster features to create discrete states.

        Args:
            n_clusters: Number of clusters (states)
            algorithm: Clustering algorithm ('kmeans', 'gmm')
        """
        logger.info(
            f"Clustering features into {n_clusters} states using {algorithm}..."
        )

        if self.features is None:
            raise ValueError("Features must be computed before clustering")

        # Adaptive microstate reduction for Stage-1: prefer 10–20 states for ~40k frames
        try:
            num_frames = int(self.features.shape[0]) if self.features is not None else 0
        except Exception:
            num_frames = 0
        if n_clusters > 20 and 0 < num_frames <= 40000:
            logger.info(
                f"Reducing requested clusters from {n_clusters} to 20 for low data volume ({num_frames} frames)"
            )
            n_clusters = 20
        if n_clusters < 10:
            n_clusters = 10

        if algorithm == "kmeans":
            clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(self.features)
            self.cluster_centers = clusterer.cluster_centers_
        else:
            raise ValueError(f"Clustering algorithm {algorithm} not implemented")

        # Split labels back into trajectories
        self.dtrajs = []
        start_idx = 0
        for traj in self.trajectories:
            end_idx = start_idx + traj.n_frames
            self.dtrajs.append(labels[start_idx:end_idx])
            start_idx = end_idx

        self.n_states = n_clusters
        logger.info(f"Clustering completed: {n_clusters} states")

    def build_msm(self, lag_time: int = 20, method: str = "standard"):
        """
        Build Markov State Model from discrete trajectories.

        Args:
            lag_time: Lag time for transition counting
            method: MSM method ('standard', 'tram')
        """
        logger.info(f"Building MSM with lag time {lag_time} using {method} method...")

        self.lag_time = lag_time
        # If features were marked for TICA (feature_type contained 'tica'), they were already projected.
        # Ensure that if features are still high-dimensional dihedrals without TICA, we optionally apply a default 3-comp TICA.
        try:
            # Use direct attribute checks to satisfy type checkers
            if self.features is not None and not hasattr(self, "tica_components_"):
                # Heuristic: if feature dimension is large (>20), project to 3 tICs
                if self.features.shape[1] > 20:
                    logger.info(
                        "Applying default 3-component TICA prior to MSM to reduce noise"
                    )
                    self._maybe_apply_tica("tica", 3)
        except Exception:
            pass

        if method == "standard":
            self._build_standard_msm(lag_time)
        elif method == "tram":
            self._build_tram_msm(lag_time)
        else:
            raise ValueError(f"Unknown MSM method: {method}")

        # Compute free energies
        self._compute_free_energies()

        logger.info("MSM construction completed")

    # ---------------- TICA (time-lagged ICA) support ----------------

    def _maybe_apply_tica(
        self, feature_type: str, n_components_hint: Optional[int]
    ) -> None:
        """Apply simple TICA if requested via feature_type string containing 'tica'.

        Projects self.features to the top 2–5 TICA components using current lag_time.
        """
        if self.features is None:
            return
        if "tica" not in feature_type.lower():
            return
        n_components = int(max(2, min(5, (n_components_hint or 3))))

        # Mean-center features
        X = self.features.astype(float)
        X -= np.mean(X, axis=0, keepdims=True)

        lag = max(1, int(self.lag_time))
        C0 = np.zeros((X.shape[1], X.shape[1]), dtype=float)
        Ctau = np.zeros_like(C0)

        # Accumulate time-lagged covariances per trajectory to avoid crossing boundaries
        start_idx = 0
        for traj in self.trajectories:
            T = traj.n_frames
            end_idx = start_idx + T
            if T > lag + 1:
                Xt = X[start_idx : end_idx - lag]
                Xtl = X[start_idx + lag : end_idx]
                C0 += Xt.T @ Xt
                Ctau += Xt.T @ Xtl
            start_idx = end_idx

        # Regularize for numerical stability
        eps = 1e-6
        C0 += eps * np.eye(C0.shape[0])

        try:
            # Solve generalized eigenproblem via inversion (small dims expected)
            A = np.linalg.solve(C0, Ctau)
            eigvals, eigvecs = np.linalg.eig(A)
            order = np.argsort(-np.abs(eigvals))
            W = np.real(eigvecs[:, order[:n_components]])
            self.features = X @ W
            # Store for reference/debugging (optional attributes)
            self.tica_components_ = W  # type: ignore[attr-defined]
            self.tica_eigenvalues_ = np.real(eigvals[order[:n_components]])  # type: ignore[attr-defined]
            logger.info(f"Applied TICA projection to {n_components} components")
        except Exception as e:
            logger.warning(f"TICA failed ({e}); proceeding without TICA")

    def _build_standard_msm(self, lag_time: int):
        """Build standard MSM from single temperature data."""
        # Count transitions with small Bayesian prior α to regularize sparse rows
        alpha: float = 2.0
        counts: Dict[Tuple[int, int], float] = defaultdict(float)
        total_transitions = 0

        for dtraj in self.dtrajs:
            for i in range(len(dtraj) - lag_time):
                state_i = dtraj[i]
                state_j = dtraj[i + lag_time]
                counts[(state_i, state_j)] += 1.0
                total_transitions += 1

        # Build count matrix
        count_matrix = np.full((self.n_states, self.n_states), alpha, dtype=float)
        for (i, j), count in counts.items():
            count_matrix[i, j] += count

        # Ensure diagonal support for completely unvisited states (in addition to α)
        row_sums_tmp = count_matrix.sum(axis=1)
        zero_row_indices = np.where(row_sums_tmp == 0)[0]
        if zero_row_indices.size > 0:
            for idx in zero_row_indices:
                count_matrix[idx, idx] = max(1.0, alpha)

        self.count_matrix = count_matrix

        # Build transition matrix (row-stochastic)
        row_sums = count_matrix.sum(axis=1)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        self.transition_matrix = (
            count_matrix / row_sums[:, np.newaxis]
        )  # Fix: Direct assignment

        # Compute stationary distribution
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        stationary_idx = np.argmax(np.real(eigenvals))
        stationary = np.real(eigenvecs[:, stationary_idx])
        stationary = stationary / stationary.sum()
        self.stationary_distribution = np.abs(stationary)  # Ensure positive

    def _build_tram_msm(self, lag_time: int):
        """Build MSM using TRAM for multi-temperature data."""
        logger.info("Building TRAM MSM for multi-temperature data...")

        # This is a simplified TRAM implementation
        # For production use, consider using packages like pyemma or deeptime

        if len(self.temperatures) == 1:
            logger.warning(
                "Only one temperature provided, falling back to standard MSM"
            )
            return self._build_standard_msm(lag_time)

        # Count transitions for each temperature - Fix: Added proper type annotation
        temp_counts: Dict[float, Dict[Tuple[int, int], float]] = (
            {}
        )  # Fix: Added type annotation
        for temp_idx, temp in enumerate(self.temperatures):
            if temp_idx < len(self.dtrajs):
                dtraj = self.dtrajs[temp_idx]
                counts: Dict[Tuple[int, int], float] = defaultdict(
                    float
                )  # Fix: Added type annotation

                for i in range(len(dtraj) - lag_time):
                    state_i = dtraj[i]
                    state_j = dtraj[i + lag_time]
                    counts[(state_i, state_j)] += 1.0

                temp_counts[temp] = counts

        # Simplified TRAM: weight by Boltzmann factors
        # This is a basic implementation - real TRAM is more sophisticated
        kT_ref = constants.k * 300.0  # Reference temperature

        combined_counts: Dict[Tuple[int, int], float] = defaultdict(
            float
        )  # Fix: Added type annotation
        for temp, counts in temp_counts.items():
            kT = constants.k * temp
            weight = kT_ref / kT  # Simple reweighting

            for (i, j), count in counts.items():
                combined_counts[(i, j)] += count * weight

        # Build matrices from combined counts with small Bayesian prior α
        alpha: float = 2.0
        count_matrix = np.full((self.n_states, self.n_states), alpha, dtype=float)
        for (i, j), count in combined_counts.items():
            count_matrix[i, j] += count

        # Ensure diagonal support for completely unvisited states (in addition to α)
        row_sums_tmp = count_matrix.sum(axis=1)
        zero_row_indices = np.where(row_sums_tmp == 0)[0]
        if zero_row_indices.size > 0:
            for idx in zero_row_indices:
                count_matrix[idx, idx] = max(1.0, alpha)

        self.count_matrix = count_matrix

        # Build transition matrix
        row_sums = count_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = (
            count_matrix / row_sums[:, np.newaxis]
        )  # Fix: Direct assignment

        # Compute stationary distribution
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        stationary_idx = np.argmax(np.real(eigenvals))
        stationary = np.real(eigenvecs[:, stationary_idx])
        stationary = stationary / stationary.sum()
        self.stationary_distribution = np.abs(stationary)

    def _compute_free_energies(self, temperature: float = 300.0):
        """Compute free energies from stationary distribution."""
        if self.stationary_distribution is None:
            raise ValueError("Stationary distribution must be computed first")

        kT = constants.k * temperature * constants.Avogadro / 1000.0  # kJ/mol

        # Avoid log(0) by adding small epsilon
        pi_safe = np.maximum(self.stationary_distribution, 1e-12)
        self.free_energies = -kT * np.log(pi_safe)

        # Set relative to minimum
        self.free_energies -= np.min(self.free_energies)

        logger.info(
            (
                "Free energies computed (range: 0 - "
                f"{np.max(self.free_energies):.2f} kJ/mol)"
            )
        )

    def compute_implied_timescales(
        self, lag_times: Optional[List[int]] = None, n_timescales: int = 5
    ):
        """
        Compute implied timescales for MSM validation.

        Args:
            lag_times: List of lag times to test
            n_timescales: Number of timescales to compute
        """
        logger.info("Computing implied timescales...")

        if lag_times is None:
            # Recommended lag grid (frames): coarse-to-fine coverage
            lag_times = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]

        timescales_data = []

        for lag in lag_times:
            try:
                # Build MSM for this lag time
                self._build_standard_msm(lag)

                # Compute eigenvalues - Fix: Ensure transition_matrix is not None
                if self.transition_matrix is not None:
                    eigenvals = np.linalg.eigvals(self.transition_matrix)
                    eigenvals = np.real(eigenvals)
                    eigenvals = np.sort(eigenvals)[::-1]  # Sort descending

                    # Convert to timescales (excluding the stationary eigenvalue)
                    timescales = []
                    for i in range(1, min(n_timescales + 1, len(eigenvals))):
                        if eigenvals[i] > 0 and eigenvals[i] < 1:
                            ts = -lag / np.log(eigenvals[i])
                            timescales.append(ts)

                    # Pad with NaN if not enough timescales
                    while len(timescales) < n_timescales:
                        timescales.append(np.nan)

                    timescales_data.append(timescales[:n_timescales])
                else:
                    timescales_data.append([np.nan] * n_timescales)

            except Exception as e:
                logger.warning(f"Failed to compute timescales for lag {lag}: {e}")
                timescales_data.append([np.nan] * n_timescales)

        self.implied_timescales = {  # Fix: Direct assignment to dict
            "lag_times": lag_times,
            "timescales": np.array(timescales_data),
        }

        # Restore original MSM
        self._build_standard_msm(self.lag_time)

        logger.info("Implied timescales computation completed")

    # ---------------- Macrostate CK test ----------------

    def _micro_to_macro_labels(self, n_macrostates: int = 3) -> Optional[np.ndarray]:
        try:
            # Prefer precomputed labels in state_table
            if (
                self.state_table is not None
                and "macrostate" in self.state_table.columns
            ):
                labels = np.asarray(self.state_table["macrostate"], dtype=int)
                if labels.size == self.n_states:
                    return labels
        except Exception:
            pass
        # Fallback: run internal lumping
        return self._pcca_lumping(n_macrostates=n_macrostates)

    def compute_ck_test_macrostates(
        self, n_macrostates: int = 3, factors: Optional[List[int]] = None
    ) -> Optional[Dict[str, float]]:
        """Compute CK test MSE at macrostate level for multiples of the base lag.

        Returns a mapping of factor -> MSE on T^factor vs empirical T at (factor*lag).
        """
        try:
            factors = self._normalize_ck_factors(factors)
            if not self.dtrajs or self.n_states <= 0 or self.lag_time <= 0:
                return None

            macro_labels = self._micro_to_macro_labels(n_macrostates=n_macrostates)
            if macro_labels is None:
                return None
            n_macros = int(np.max(macro_labels) + 1)
            if n_macros <= 1:
                return None

            macro_trajs = self._build_macro_trajectories(self.dtrajs, macro_labels)
            T1 = self._estimate_macro_T(macro_trajs, n_macros, int(self.lag_time))

            results: Dict[str, float] = {}
            for f in factors:
                mse = self._ck_mse_for_factor(
                    T1, macro_trajs, n_macros, int(self.lag_time), int(f)
                )
                if mse is not None:
                    results[str(int(f))] = float(mse)

            self._persist_ck_macro_results(results, n_macros, int(self.lag_time))
            return results
        except Exception:
            return None

    # ---- CK helpers (split to address C901) ----

    def _normalize_ck_factors(self, factors: Optional[List[int]]) -> List[int]:
        if factors is None:
            return [2, 3]
        return [int(f) for f in factors if int(f) > 1]

    def _build_macro_trajectories(
        self, dtrajs: List[np.ndarray], macro_labels: np.ndarray
    ) -> List[np.ndarray]:
        macro_trajs: List[np.ndarray] = []
        for arr in dtrajs:
            try:
                seq = np.asarray(arr, dtype=int)
                macro_trajs.append(macro_labels[seq])
            except Exception:
                macro_trajs.append(np.array([], dtype=int))
        return macro_trajs

    def _count_macro_T(
        self, macro_trajs: List[np.ndarray], nM: int, lag: int
    ) -> np.ndarray:
        C = np.zeros((nM, nM), dtype=float)
        for seq in macro_trajs:
            if seq.size <= lag:
                continue
            for i in range(0, seq.size - lag):
                a = int(seq[i])
                b = int(seq[i + lag])
                if 0 <= a < nM and 0 <= b < nM:
                    C[a, b] += 1.0
        rows = C.sum(axis=1)
        rows[rows == 0] = 1.0
        return C / rows[:, None]

    def _estimate_macro_T(
        self, macro_trajs: List[np.ndarray], nM: int, lag_frames: int
    ) -> np.ndarray:
        return self._count_macro_T(macro_trajs, nM, int(lag_frames))

    def _ck_mse_for_factor(
        self,
        T1: np.ndarray,
        macro_trajs: List[np.ndarray],
        nM: int,
        base_lag: int,
        factor: int,
    ) -> Optional[float]:
        try:
            T_theory = np.linalg.matrix_power(T1, int(factor))
            T_emp = self._count_macro_T(macro_trajs, nM, int(base_lag) * int(factor))
            diff = T_theory - T_emp
            return float(np.mean(diff * diff))
        except Exception:
            return None

    def _persist_ck_macro_results(
        self, results: Dict[str, float], nM: int, lag_frames: int
    ) -> None:
        try:
            out = {
                "n_macrostates": int(nM),
                "lag_time_frames": int(lag_frames),
                "factors": results,
            }
            with open(
                self.output_dir / "msm_analysis_ck_macro.json", "w", encoding="utf-8"
            ) as f:
                json.dump(out, f, indent=2)
        except Exception:
            # Best-effort persistence; do not raise
            pass

    def generate_free_energy_surface(
        self,
        cv1_name: str = "phi",
        cv2_name: str = "psi",
        bins: int = 50,
        temperature: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Generate 2D free energy surface from MSM data.

        Args:
            cv1_name: Name of first collective variable
            cv2_name: Name of second collective variable
            bins: Number of bins for 2D histogram
            temperature: Temperature for free energy calculation

        Returns:
            Dictionary containing FES data
        """
        logger.info(f"Generating free energy surface: {cv1_name} vs {cv2_name}")
        self._validate_fes_prerequisites()
        cv1_data, cv2_data = self._extract_collective_variables(cv1_name, cv2_name)
        frame_weights_array = self._map_stationary_to_frame_weights()
        total_frames, cv_points = self._log_data_cardinality(
            cv1_data, frame_weights_array
        )
        bins = self._choose_bins(total_frames, bins)
        cv1_data, cv2_data, frame_weights_array = self._align_data_lengths(
            cv1_data, cv2_data, frame_weights_array
        )
        ranges = (
            [(-180.0, 180.0), (-180.0, 180.0)]
            if (cv1_name == "phi" and cv2_name == "psi")
            else None
        )
        H, xedges, yedges = self._compute_weighted_histogram(
            cv1_data, cv2_data, frame_weights_array, bins, ranges, smooth_sigma=0.6
        )
        F = self._histogram_to_free_energy(H, temperature)
        self._log_fes_statistics(F, H)
        self._store_fes_result(F, xedges, yedges, cv1_name, cv2_name, temperature)
        try:
            # Side-by-side scatter and FES for quick sanity diagnostics
            self._save_fes_with_scatter(
                cv1_data, cv2_data, xedges, yedges, F, cv1_name, cv2_name
            )
        except Exception:
            pass
        logger.info("Free energy surface generated")
        assert self.fes_data is not None
        return self.fes_data

    # ---------------- FES helper methods (split for C901) ----------------

    def _validate_fes_prerequisites(self) -> None:
        if self.features is None or self.stationary_distribution is None:
            raise ValueError("Features and MSM must be computed first")

    def _map_stationary_to_frame_weights(self) -> np.ndarray:
        frame_weights: list[float] = []
        station = self.stationary_distribution
        if station is None:
            raise ValueError("Stationary distribution not available")
        for dtraj in self.dtrajs:
            for state in dtraj:
                frame_weights.append(float(station[state]))
        return np.array(frame_weights)

    def _log_data_cardinality(
        self, cv1_data: np.ndarray, frame_weights_array: np.ndarray
    ) -> tuple[int, int]:
        total_frames = int(len(frame_weights_array))
        cv_points = int(len(cv1_data))
        logger.info(
            (
                "MSM Analysis data: "
                f"{cv_points} CV points, {total_frames} trajectory frames"
            )
        )
        return total_frames, cv_points

    def _choose_bins(self, total_frames: int, user_bins: int) -> int:
        """Select bin count based on frames; clamp to [40, 60]."""
        # Simple heuristic: scale with sqrt of frames but clamp to 40–60
        try:
            if total_frames <= 0:
                return max(40, min(60, user_bins))
            reco = int(max(40, min(60, np.sqrt(total_frames) // 6)))
        except Exception:
            reco = 50
        # Respect user suggestion if within clamp, otherwise prefer recommended
        candidate = max(40, min(60, int(user_bins)))
        # Pick the closer to recommended to avoid extremes
        return candidate if abs(candidate - reco) <= 5 else reco

    def _align_data_lengths(
        self,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_length = min(len(cv1_data), len(cv2_data), len(frame_weights_array))
        if len(cv1_data) != len(frame_weights_array):
            logger.warning(
                (
                    f"Length mismatch: CV data ({len(cv1_data)}) vs weights "
                    f"({len(frame_weights_array)}). Truncating to {min_length} "
                    "points."
                )
            )
            cv1_data = cv1_data[:min_length]
            cv2_data = cv2_data[:min_length]
            frame_weights_array = frame_weights_array[:min_length]
        return cv1_data, cv2_data, frame_weights_array

    def _compute_weighted_histogram(
        self,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
        bins: int,
        ranges: Optional[List[Tuple[float, float]]] = None,
        smooth_sigma: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            H, xedges, yedges = np.histogram2d(
                cv1_data,
                cv2_data,
                bins=bins,
                weights=frame_weights_array,
                density=True,
                range=ranges,  # Use fixed ranges when provided (e.g., [-180,180] for angles)
            )
            if smooth_sigma and smooth_sigma > 0:
                H = gaussian_filter(H, sigma=float(smooth_sigma))
            non_zero_bins = int(np.sum(H > 0))
            logger.info(
                (
                    "Histogram: "
                    f"{non_zero_bins} non-zero bins out of {bins*bins} total"
                )
            )
            if non_zero_bins < 3:
                logger.warning("Very sparse histogram - results may not be meaningful")
            return H, xedges, yedges
        except Exception as e:
            logger.error(f"Histogram generation failed: {e}")
            raise ValueError(f"Could not generate histogram for FES: {e}")

    def _histogram_to_free_energy(
        self, H: np.ndarray, temperature: float
    ) -> np.ndarray:
        kT = constants.k * temperature * constants.Avogadro / 1000.0
        F = np.full_like(H, np.inf)
        mask = H > 1e-12
        if int(np.sum(mask)) == 0:
            logger.error(
                ("No populated bins in histogram - cannot generate " "meaningful FES")
            )
            raise ValueError(
                "Histogram too sparse for free energy calculation. "
                "Try: 1) Longer simulation, 2) Fewer bins, 3) Different CVs"
            )
        F[mask] = -kT * np.log(H[mask])
        finite_mask = np.isfinite(F)
        if int(np.sum(finite_mask)) == 0:
            logger.error("No finite free energy values - calculation failed")
            raise ValueError(
                "All free energy values are infinite - histogram too sparse"
            )
        F_min = float(np.min(F[finite_mask]))
        F[finite_mask] -= F_min
        return F

    def _log_fes_statistics(self, F: np.ndarray, H: np.ndarray) -> None:
        n_finite = int(np.sum(np.isfinite(F)))
        n_total = int(H.size)
        F_min = float(np.min(F[np.isfinite(F)]))
        F_max = float(np.max(F[np.isfinite(F)]))
        logger.info(
            (
                f"Free energy surface: {n_finite}/{n_total} finite bins, "
                f"range: {F_min:.2f} to {F_max:.2f} kJ/mol"
            )
        )

    def _store_fes_result(
        self,
        F: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        cv1_name: str,
        cv2_name: str,
        temperature: float,
    ) -> None:
        self.fes_data = {
            "free_energy": F,
            "xedges": xedges,
            "yedges": yedges,
            "cv1_name": cv1_name,
            "cv2_name": cv2_name,
            "temperature": temperature,
        }

    def _save_fes_with_scatter(
        self,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        F: np.ndarray,
        cv1_name: str,
        cv2_name: str,
    ) -> None:
        # Downsample scatter for speed/clarity
        max_points = 20000
        n = min(len(cv1_data), len(cv2_data))
        if n <= 0:
            return
        stride = max(1, n // max_points)
        xs = cv1_data[::stride]
        ys = cv2_data[::stride]

        import matplotlib.pyplot as _plt

        fig, axes = _plt.subplots(1, 2, figsize=(12, 5))
        # Scatter
        axes[0].scatter(xs, ys, s=4, alpha=0.25, c="tab:blue")
        axes[0].set_xlim([-180, 180])
        axes[0].set_ylim([-180, 180])
        axes[0].set_xlabel(f"{cv1_name} (deg)")
        axes[0].set_ylabel(f"{cv2_name} (deg)")
        axes[0].set_title("Raw scatter")
        # FES contour
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])
        c = axes[1].contourf(x_centers, y_centers, F.T, levels=20, cmap="viridis")
        fig.colorbar(c, ax=axes[1], label="Free Energy (kJ/mol)")
        axes[1].set_xlabel(cv1_name)
        axes[1].set_ylabel(cv2_name)
        axes[1].set_title("FES (smoothed)")
        fig.suptitle(f"{cv1_name} vs {cv2_name}")
        fig.tight_layout()
        fig.savefig(self.output_dir / "fes_and_scatter.png", dpi=200)
        _plt.close(fig)

    def _extract_collective_variables(
        self, cv1_name: str, cv2_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract collective variables from trajectory data."""
        cv1_data: list[float] = []
        cv2_data: list[float] = []

        for traj in self.trajectories:
            if cv1_name == "phi" and cv2_name == "psi":
                phi_angles, _ = md.compute_phi(traj)
                psi_angles, _ = md.compute_psi(traj)

                # Ensure arrays are 1D by selecting the first available dihedral
                if phi_angles.size > 0 and psi_angles.size > 0:
                    phi_vec = phi_angles[:, 0] if phi_angles.ndim == 2 else phi_angles
                    psi_vec = psi_angles[:, 0] if psi_angles.ndim == 2 else psi_angles
                    # Convert radians -> degrees and wrap to [-180, 180]
                    phi_deg = np.degrees(np.array(phi_vec).reshape(-1))
                    psi_deg = np.degrees(np.array(psi_vec).reshape(-1))
                    phi_wrapped = ((phi_deg + 180.0) % 360.0) - 180.0
                    psi_wrapped = ((psi_deg + 180.0) % 360.0) - 180.0
                    cv1_data.extend([float(v) for v in phi_wrapped])
                    cv2_data.extend([float(v) for v in psi_wrapped])
                else:
                    raise ValueError("No phi/psi angles found in trajectory")

            elif "distance" in cv1_name or "distance" in cv2_name:
                # Simple distance features
                ca_indices = traj.topology.select("name CA")
                if len(ca_indices) >= 4:
                    dist1 = md.compute_distances(
                        traj, [[ca_indices[0], ca_indices[-1]]]
                    )
                    dist2 = md.compute_distances(
                        traj, [[ca_indices[len(ca_indices) // 2], ca_indices[-1]]]
                    )
                    cv1_data.extend(dist1.flatten())
                    cv2_data.extend(dist2.flatten())
                else:
                    raise ValueError("Insufficient atoms for distance calculation")

            else:
                # Use first two principal components of features
                if self.features is None:
                    raise ValueError("Features not computed")
                if self.features.shape[1] >= 2:
                    start_idx = sum(
                        t.n_frames
                        for t in self.trajectories[: self.trajectories.index(traj)]
                    )
                    end_idx = start_idx + traj.n_frames
                    cv1_data.extend(self.features[start_idx:end_idx, 0])
                    cv2_data.extend(self.features[start_idx:end_idx, 1])
                else:
                    raise ValueError("Insufficient feature dimensions")

        return np.array(cv1_data, dtype=float), np.array(cv2_data, dtype=float)

    def create_state_table(self) -> pd.DataFrame:
        """Create comprehensive state summary table."""
        logger.info("Creating state summary table...")
        self._validate_state_table_prerequisites()
        state_data = self._build_basic_state_info()
        frame_counts, total_frames = self._count_frames_per_state()
        state_data["frame_count"] = frame_counts.astype(int)
        state_data["frame_percentage"] = 100 * frame_counts / max(total_frames, 1)
        representative_frames, centroid_features = self._find_representatives()
        rep_traj_array, rep_frame_array = self._representative_arrays(
            representative_frames
        )
        state_data["representative_traj"] = rep_traj_array
        state_data["representative_frame"] = rep_frame_array
        self._attach_cluster_centers(state_data)
        self.state_table = pd.DataFrame(state_data)
        logger.info(f"State table created with {len(self.state_table)} states")
        try:
            # Optional: PCCA+ lumping from microstates to macrostates (3–5)
            macrostates = self._pcca_lumping(n_macrostates=4)
            if macrostates is not None:
                self.state_table["macrostate"] = macrostates
        except Exception as e:
            logger.warning(f"PCCA+ lumping skipped ({e})")
        return self.state_table

    # -------------- State table helper methods (split for C901) --------------

    def _validate_state_table_prerequisites(self) -> None:
        if self.stationary_distribution is None:
            raise ValueError("MSM must be built before creating state table")

    def _build_basic_state_info(self) -> Dict[str, Any]:
        return {
            "state_id": range(self.n_states),
            "population": self.stationary_distribution,
            "free_energy_kJ_mol": (
                self.free_energies
                if self.free_energies is not None
                else np.zeros(self.n_states)
            ),
            "free_energy_kcal_mol": (
                self.free_energies * 0.239006
                if self.free_energies is not None
                else np.zeros(self.n_states)
            ),
        }

    def _count_frames_per_state(self) -> tuple[np.ndarray, int]:
        frame_counts = np.zeros(self.n_states)
        total_frames = 0
        for dtraj in self.dtrajs:
            for state in dtraj:
                frame_counts[state] += 1
                total_frames += 1
        return frame_counts, total_frames

    def _find_representatives(
        self,
    ) -> tuple[list[tuple[int, int]], list[Optional[np.ndarray]]]:
        representative_frames: list[tuple[int, int]] = []
        centroid_features: list[Optional[np.ndarray]] = []
        for state in range(self.n_states):
            state_frames: list[tuple[int, int]] = []
            state_features: list[np.ndarray] = []
            frame_idx = 0
            for traj_idx, dtraj in enumerate(self.dtrajs):
                for _local_frame, assigned_state in enumerate(dtraj):
                    if assigned_state == state:
                        state_frames.append((traj_idx, _local_frame))
                        if self.features is not None:
                            state_features.append(self.features[frame_idx])
                    frame_idx += 1
            if state_features:
                state_features_array = np.array(state_features)
                centroid = np.mean(state_features_array, axis=0)
                distances = np.linalg.norm(state_features_array - centroid, axis=1)
                closest_idx = int(np.argmin(distances))
                representative_frames.append(state_frames[closest_idx])
                centroid_features.append(centroid)
            else:
                representative_frames.append((-1, -1))
                centroid_features.append(None)
        return representative_frames, centroid_features

    def _representative_arrays(
        self, representative_frames: list[tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        rep_traj_array = np.array([int(rf[0]) for rf in representative_frames])
        rep_frame_array = np.array([int(rf[1]) for rf in representative_frames])
        return rep_traj_array, rep_frame_array

    def _attach_cluster_centers(self, state_data: Dict[str, Any]) -> None:
        if self.cluster_centers is not None:
            for i, center in enumerate(self.cluster_centers.T):
                state_data[f"cluster_center_{i}"] = center

    def _pcca_lumping(self, n_macrostates: int = 4) -> Optional[np.ndarray]:
        """Perform a simple PCCA+-like lumping using leading eigenvectors.

        This is a lightweight surrogate: we compute the top-k right eigenvectors of
        T and run k-means in that space to get fuzzy clusters; then assign hard labels.
        """
        try:
            if self.transition_matrix is None or self.n_states <= n_macrostates:
                return None
            T = np.asarray(self.transition_matrix, dtype=float)
            # Leading eigenvectors (excluding stationary)
            eigvals, eigvecs = np.linalg.eig(T.T)
            order = np.argsort(-np.real(eigvals))
            # Skip the first (stationary); take next components
            k = max(2, min(n_macrostates, T.shape[0] - 1))
            comps = np.real(eigvecs[:, order[1 : 1 + k]])
            # k-means in eigenvector space
            km = MiniBatchKMeans(n_clusters=n_macrostates, random_state=42)
            labels = km.fit_predict(comps)
            return labels.astype(int)
        except Exception:
            return None

    def _create_matrix_intelligent(
        self, shape: Tuple[int, int], use_sparse: Optional[bool] = None
    ) -> Union[np.ndarray, csc_matrix]:
        """
        Intelligently create matrix (sparse or dense) based on expected size and sparsity.

        Args:
            shape: Matrix shape (n_states, n_states)
            use_sparse: Force sparse (True) or dense (False). If None, auto-decide.

        Returns:
            Zero matrix of appropriate type
        """
        n_states = shape[0]

        if use_sparse is None:
            # Auto-decide based on size - sparse for large state spaces
            use_sparse = n_states > 100

        if use_sparse:
            logger.debug(f"Creating sparse matrix ({n_states}x{n_states})")
            return csc_matrix(shape, dtype=np.float64)
        else:
            logger.debug(f"Creating dense matrix ({n_states}x{n_states})")
            return np.zeros(shape, dtype=np.float64)

    def _matrix_add_count(
        self, matrix: Union[np.ndarray, csc_matrix], i: int, j: int, count: float
    ):
        """Add count to matrix element, handling both sparse and dense matrices."""
        if issparse(matrix):
            matrix[i, j] += count
        else:
            matrix[i, j] += count

    def _matrix_normalize_rows(
        self, matrix: Union[np.ndarray, csc_matrix]
    ) -> Union[np.ndarray, csc_matrix]:
        """Normalize matrix rows to create transition matrix, handling both sparse and dense."""
        if issparse(matrix):
            # Sparse matrix row normalization
            row_sums = np.array(matrix.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            row_diag = csc_matrix(
                (1.0 / row_sums, (range(len(row_sums)), range(len(row_sums)))),
                shape=(len(row_sums), len(row_sums)),
            )
            return row_diag @ matrix
        else:
            # Dense matrix row normalization
            row_sums = matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1
            return matrix / row_sums[:, np.newaxis]

    def _save_matrix_intelligent(
        self, matrix, filename_base: str, prefix: str = "msm_analysis"
    ):
        """
        Intelligently save matrix in appropriate format(s) based on type and size.

        Args:
            matrix: numpy array or scipy sparse matrix
            filename_base: base filename (e.g., "transition_matrix")
            prefix: file prefix
        """
        if matrix is None:
            return

        # Always save dense format for compatibility
        np.save(
            self.output_dir / f"{prefix}_{filename_base}.npy",
            matrix.toarray() if issparse(matrix) else matrix,
        )

        # For large matrices, also save sparse format for efficiency
        if matrix.size > 10000:  # >100x100 matrix
            if issparse(matrix):
                # Already sparse - save directly
                save_npz(self.output_dir / f"{prefix}_{filename_base}.npz", matrix)
            else:
                # Convert dense to sparse if beneficial
                # Only convert if matrix is actually sparse (>95% zeros)
                sparsity = np.count_nonzero(matrix) / matrix.size
                if sparsity < 0.05:  # Less than 5% non-zero
                    sparse_matrix = csc_matrix(matrix)
                    save_npz(
                        self.output_dir / f"{prefix}_{filename_base}.npz", sparse_matrix
                    )
                    logger.info(
                        (
                            f"Converted {filename_base} to sparse format (sparsity: "
                            f"{(1-sparsity)*100:.1f}% zeros)"
                        )
                    )
                else:
                    logger.debug(
                        (
                            f"Keeping {filename_base} as dense (sparsity: "
                            f"{(1-sparsity)*100:.1f}% zeros)"
                        )
                    )

    def save_analysis_results(self, prefix: str = "msm_analysis"):
        """
        Save all analysis results to files.

        This method orchestrates saving all artifacts by delegating to smaller
        helpers to reduce complexity while preserving exact behavior.
        """
        logger.info("Saving analysis results...")

        # Core matrices
        self._save_transition_matrix(prefix)
        self._save_count_matrix(prefix)

        # Scalars and distributions
        self._save_free_energies(prefix)
        self._save_stationary_distribution(prefix)

        # Trajectories and tables
        self._save_discrete_trajectories(prefix)
        self._save_state_table_file(prefix)

        # FES and metadata
        self._save_fes_array(prefix)
        self._save_fes_metadata(prefix)

        # Analysis artifacts
        self._save_implied_timescales_file(prefix)

        # Final message
        self._log_save_completion()

    # ---------------- Save helpers (split to address C901) ----------------

    def _save_transition_matrix(self, prefix: str) -> None:
        """Save the transition matrix using intelligent format selection."""
        self._save_matrix_intelligent(
            self.transition_matrix, "transition_matrix", prefix
        )

    def _save_count_matrix(self, prefix: str) -> None:
        """Save the count matrix using intelligent format selection."""
        self._save_matrix_intelligent(self.count_matrix, "count_matrix", prefix)

    def _save_free_energies(self, prefix: str) -> None:
        """Save free energies if available."""
        if self.free_energies is not None:
            np.save(
                self.output_dir / f"{prefix}_free_energies.npy",
                self.free_energies,
            )

    def _save_stationary_distribution(self, prefix: str) -> None:
        """Save stationary distribution if available."""
        if self.stationary_distribution is not None:
            np.save(
                self.output_dir / f"{prefix}_stationary_distribution.npy",
                self.stationary_distribution,
            )

    def _save_discrete_trajectories(self, prefix: str) -> None:
        """Save discrete trajectories with object-array optimization and fallback."""
        if not self.dtrajs:
            return
        try:
            dtrajs_obj = self._convert_dtrajs_to_object_array()
            np.save(self.output_dir / f"{prefix}_dtrajs.npy", dtrajs_obj)
        except Exception:
            self._save_dtrajs_individually(prefix)

    def _convert_dtrajs_to_object_array(self) -> np.ndarray:
        """Convert list of variable-length trajectories to an object ndarray."""
        return np.array(self.dtrajs, dtype=object)

    def _save_dtrajs_individually(self, prefix: str) -> None:
        """Fallback saver for discrete trajectories, each as a separate file."""
        for idx, dtraj in enumerate(self.dtrajs):
            try:
                np.save(
                    self.output_dir / f"{prefix}_dtrajs_traj{idx:02d}.npy",
                    np.asarray(dtraj),
                )
            except Exception:
                continue

    def _save_state_table_file(self, prefix: str) -> None:
        """Save state table to CSV if available."""
        if self.state_table is not None:
            self.state_table.to_csv(
                self.output_dir / f"{prefix}_state_table.csv",
                index=False,
            )

    def _save_fes_array(self, prefix: str) -> None:
        """Save free energy surface array if present."""
        if self.fes_data is None:
            return
        np.save(
            self.output_dir / f"{prefix}_fes.npy",
            self.fes_data["free_energy"],
        )

    def _save_fes_metadata(self, prefix: str) -> None:
        """Save non-array FES metadata alongside the FES array."""
        if self.fes_data is None:
            return
        fes_metadata = {k: v for k, v in self.fes_data.items() if k != "free_energy"}
        with open(self.output_dir / f"{prefix}_fes_metadata.pkl", "wb") as f:
            pickle.dump(fes_metadata, f)

    def _save_implied_timescales_file(self, prefix: str) -> None:
        """Save implied timescales structure if computed."""
        if self.implied_timescales is None:
            return
        with open(self.output_dir / f"{prefix}_implied_timescales.pkl", "wb") as f:
            pickle.dump(self.implied_timescales, f)

    def _log_save_completion(self) -> None:
        """Emit a final log confirming save destination."""
        logger.info(f"Analysis results saved to {self.output_dir}")

    def plot_free_energy_surface(
        self, save_file: Optional[str] = None, interactive: bool = False
    ):
        """Plot the free energy surface."""
        if self.fes_data is None:
            raise ValueError("Free energy surface must be generated first")

        F = self.fes_data["free_energy"]
        xedges = self.fes_data["xedges"]
        yedges = self.fes_data["yedges"]
        cv1_name = self.fes_data["cv1_name"]
        cv2_name = self.fes_data["cv2_name"]

        if interactive:
            try:
                import plotly.graph_objects as go

                # Create interactive plot
                x_centers = 0.5 * (xedges[:-1] + xedges[1:])
                y_centers = 0.5 * (yedges[:-1] + yedges[1:])

                fig = go.Figure(
                    data=go.Contour(
                        z=F.T,
                        x=x_centers,
                        y=y_centers,
                        colorscale="viridis",
                        colorbar=dict(title="Free Energy (kJ/mol)"),
                    )
                )

                fig.update_layout(
                    title=f"Free Energy Surface ({cv1_name} vs {cv2_name})",
                    xaxis_title=cv1_name,
                    yaxis_title=cv2_name,
                )

                if save_file:
                    fig.write_html(str(self.output_dir / f"{save_file}.html"))

                fig.show()

            except ImportError:
                logger.warning("Plotly not available, falling back to matplotlib")
                interactive = False

        if not interactive:
            # Create matplotlib plot
            plt.figure(figsize=(10, 8))

            x_centers = 0.5 * (xedges[:-1] + xedges[1:])
            y_centers = 0.5 * (yedges[:-1] + yedges[1:])

            contour = plt.contourf(x_centers, y_centers, F.T, levels=20, cmap="viridis")
            plt.colorbar(contour, label="Free Energy (kJ/mol)")

            plt.xlabel(cv1_name)
            plt.ylabel(cv2_name)
            plt.title(f"Free Energy Surface ({cv1_name} vs {cv2_name})")

            if save_file:
                plt.savefig(
                    self.output_dir / f"{save_file}.png", dpi=300, bbox_inches="tight"
                )

            plt.show()

    def plot_implied_timescales(self, save_file: Optional[str] = None):
        """Plot implied timescales for MSM validation."""
        if self.implied_timescales is None:
            raise ValueError("Implied timescales must be computed first")

        lag_times = self.implied_timescales["lag_times"]
        timescales = self.implied_timescales["timescales"]

        plt.figure(figsize=(10, 6))

        for i in range(timescales.shape[1]):
            plt.plot(lag_times, timescales[:, i], "o-", label=f"Timescale {i+1}")

        plt.xlabel("Lag Time")
        plt.ylabel("Implied Timescale")
        plt.title("Implied Timescales Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_file:
            plt.savefig(
                self.output_dir / f"{save_file}.png", dpi=300, bbox_inches="tight"
            )

        plt.show()

    def plot_free_energy_profile(self, save_file: Optional[str] = None):
        """Plot 1D free energy profile by state."""
        if self.free_energies is None:
            raise ValueError("Free energies must be computed first")

        plt.figure(figsize=(12, 6))

        state_ids = np.arange(len(self.free_energies))
        plt.bar(state_ids, self.free_energies, alpha=0.7, color="steelblue")

        plt.xlabel("State Index")
        plt.ylabel("Free Energy (kJ/mol)")
        plt.title("Free Energy Profile by State")
        plt.grid(True, alpha=0.3)

        if save_file:
            plt.savefig(
                self.output_dir / f"{save_file}.png", dpi=300, bbox_inches="tight"
            )

        plt.show()

    def extract_representative_structures(self, save_pdb: bool = True):
        """Extract and optionally save representative structures for each state."""
        logger.info("Extracting representative structures...")

        if self.state_table is None:
            self.create_state_table()

        representative_structures = []

        # Fix: Ensure state_table is not None before iterating
        if self.state_table is not None:
            for _, row in self.state_table.iterrows():
                try:
                    traj_idx = int(row["representative_traj"])
                    frame_idx = int(row["representative_frame"])
                    state_id = int(row["state_id"])

                    if traj_idx >= 0 and frame_idx >= 0:
                        # Validate indices
                        if traj_idx >= len(self.trajectories):
                            logger.warning(
                                f"Invalid trajectory index {traj_idx} for state {state_id}"
                            )
                            continue

                        traj = self.trajectories[traj_idx]
                        if frame_idx >= len(traj):
                            logger.warning(
                                f"Invalid frame index {frame_idx} for state {state_id}"
                            )
                            continue

                        # Extract frame
                        frame = traj[frame_idx]
                        representative_structures.append((state_id, frame))

                        if save_pdb:
                            output_file = (
                                self.output_dir
                                / f"state_{state_id:03d}_representative.pdb"
                            )
                            frame.save_pdb(str(output_file))

                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(
                        f"Error extracting representative structure for state {state_id}: {e}"
                    )
                    continue

        logger.info(
            f"Extracted {len(representative_structures)} representative structures"
        )
        return representative_structures


# Convenience function for complete analysis pipeline
def run_complete_msm_analysis(
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    output_dir: str = "output/msm_analysis",
    n_clusters: int = 100,
    lag_time: int = 20,
    feature_type: str = "phi_psi",
    temperatures: Optional[List[float]] = None,
) -> EnhancedMSM:
    """
    Run complete MSM analysis pipeline.

    Args:
        trajectory_files: Trajectory file(s) to analyze
        topology_file: Topology file (PDB)
        output_dir: Output directory
        n_clusters: Number of states for clustering
        lag_time: Lag time for MSM construction
        feature_type: Type of features to use
        temperatures: Temperatures for TRAM analysis

    Returns:
        EnhancedMSM object with completed analysis
    """
    msm = _initialize_msm_analyzer(
        trajectory_files, topology_file, temperatures, output_dir
    )
    _load_and_prepare(msm, feature_type, n_clusters)
    _build_and_validate_msm(msm, temperatures, lag_time)
    fes_success = _maybe_generate_fes(msm, feature_type)
    _finalize_results_and_plots(msm, fes_success)
    logger.info("Complete MSM analysis finished")
    return msm


# ---------------- Pipeline helper functions (split for C901) ----------------


def _initialize_msm_analyzer(
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    temperatures: Optional[List[float]],
    output_dir: str,
) -> EnhancedMSM:
    return EnhancedMSM(
        trajectory_files=trajectory_files,
        topology_file=topology_file,
        temperatures=temperatures,
        output_dir=output_dir,
    )


def _load_and_prepare(msm: EnhancedMSM, feature_type: str, n_clusters: int) -> None:
    msm.load_trajectories()
    # Save a quick φ/ψ sanity scatter before feature building (helps spot issues early)
    try:
        msm.save_phi_psi_scatter_diagnostics()
    except Exception:
        pass
    msm.compute_features(feature_type=feature_type)
    msm.cluster_features(n_clusters=n_clusters)


def _build_and_validate_msm(
    msm: EnhancedMSM, temperatures: Optional[List[float]], lag_time: int
) -> None:
    method = "tram" if temperatures and len(temperatures) > 1 else "standard"
    msm.build_msm(lag_time=lag_time, method=method)
    msm.compute_implied_timescales()


def _maybe_generate_fes(msm: EnhancedMSM, feature_type: str) -> bool:
    success = False
    try:
        if feature_type == "phi_psi":
            msm.generate_free_energy_surface(cv1_name="phi", cv2_name="psi")
        else:
            msm.generate_free_energy_surface(cv1_name="CV1", cv2_name="CV2")
        success = True
        logger.info("\u2713 Free energy surface generation completed")
    except ValueError as e:
        logger.warning(f"\u26a0 Free energy surface generation failed: {e}")
        logger.info("Continuing with analysis without FES plots...")
    return success


def _finalize_results_and_plots(msm: EnhancedMSM, fes_success: bool) -> None:
    msm.create_state_table()
    msm.extract_representative_structures()
    msm.save_analysis_results()
    if fes_success:
        try:
            msm.plot_free_energy_surface(save_file="free_energy_surface")
            logger.info("\u2713 Free energy surface plot saved")
        except Exception as e:
            logger.warning(f"\u26a0 Free energy surface plotting failed: {e}")
    else:
        logger.info(
            "\u26a0 Skipping free energy surface plots due to insufficient data"
        )
    try:
        msm.plot_implied_timescales(save_file="implied_timescales")
        logger.info("\u2713 Implied timescales plot saved")
    except Exception as e:
        logger.warning(f"\u26a0 Implied timescales plotting failed: {e}")
    try:
        msm.plot_free_energy_profile(save_file="free_energy_profile")
        logger.info("\u2713 Free energy profile plot saved")
    except Exception as e:
        logger.warning(f"\u26a0 Free energy profile plotting failed: {e}")
