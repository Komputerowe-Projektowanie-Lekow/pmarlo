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

    # ---------------- Feature computation helper methods ----------------

    def _log_compute_features_start(self, feature_type: str) -> None:
        logger.info(f"Computing {feature_type} features...")

    def _compute_features_for_traj(
        self, traj: md.Trajectory, feature_type: str, n_features: Optional[int]
    ) -> np.ndarray:
        if feature_type == "phi_psi":
            return self._compute_phi_psi_features(traj)
        if feature_type == "distances":
            return self._compute_distance_features(traj, n_features)
        if feature_type == "contacts":
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

        if method == "standard":
            self._build_standard_msm(lag_time)
        elif method == "tram":
            self._build_tram_msm(lag_time)
        else:
            raise ValueError(f"Unknown MSM method: {method}")

        # Compute free energies
        self._compute_free_energies()

        logger.info("MSM construction completed")

    def _build_standard_msm(self, lag_time: int):
        """Build standard MSM from single temperature data."""
        # Count transitions - Fix: Added proper type annotation
        counts: Dict[Tuple[int, int], float] = defaultdict(
            float
        )  # Fix: Added type annotation
        total_transitions = 0

        for dtraj in self.dtrajs:
            for i in range(len(dtraj) - lag_time):
                state_i = dtraj[i]
                state_j = dtraj[i + lag_time]
                counts[(state_i, state_j)] += 1.0
                total_transitions += 1

        # Build count matrix - Fix: Properly initialize as numpy array
        count_matrix = np.zeros(
            (self.n_states, self.n_states)
        )  # Fix: Direct numpy array initialization
        for (i, j), count in counts.items():
            count_matrix[i, j] = count

        # Regularize zero rows to avoid non-stochastic rows (unvisited states)
        row_sums_tmp = count_matrix.sum(axis=1)
        zero_row_indices = np.where(row_sums_tmp == 0)[0]
        if zero_row_indices.size > 0:
            for idx in zero_row_indices:
                count_matrix[idx, idx] = 1.0

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

        # Build matrices from combined counts - Fix: Proper numpy array initialization
        count_matrix = np.zeros(
            (self.n_states, self.n_states)
        )  # Fix: Direct numpy array initialization
        for (i, j), count in combined_counts.items():
            count_matrix[i, j] = count

        # Regularize zero rows to avoid non-stochastic rows (unvisited states)
        row_sums_tmp = count_matrix.sum(axis=1)
        zero_row_indices = np.where(row_sums_tmp == 0)[0]
        if zero_row_indices.size > 0:
            for idx in zero_row_indices:
                count_matrix[idx, idx] = 1.0

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
            lag_times = list(range(1, 101, 5))  # 1 to 100 in steps of 5

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
        bins = self._maybe_adjust_bins_for_sparsity(bins, total_frames)
        cv1_data, cv2_data, frame_weights_array = self._align_data_lengths(
            cv1_data, cv2_data, frame_weights_array
        )
        H, xedges, yedges = self._compute_weighted_histogram(
            cv1_data, cv2_data, frame_weights_array, bins
        )
        F = self._histogram_to_free_energy(H, temperature)
        self._log_fes_statistics(F, H)
        self._store_fes_result(F, xedges, yedges, cv1_name, cv2_name, temperature)
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

    def _maybe_adjust_bins_for_sparsity(self, bins: int, total_frames: int) -> int:
        min_frames_required = bins // 4
        if total_frames < min_frames_required:
            logger.warning(
                (
                    f"Insufficient data for {bins}x{bins} FES: "
                    f"{total_frames} frames < {min_frames_required} required"
                )
            )
            logger.info(
                (
                    "Reducing bins from "
                    f"{bins} to {max(10, total_frames // 2)} for sparse data"
                )
            )
            return max(10, total_frames // 2)
        return bins

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            H, xedges, yedges = np.histogram2d(
                cv1_data, cv2_data, bins=bins, weights=frame_weights_array, density=True
            )
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

                # Ensure arrays are 1D by selecting the first available dihedral or fallback
                if phi_angles.size > 0 and psi_angles.size > 0:
                    phi_vec = phi_angles[:, 0] if phi_angles.ndim == 2 else phi_angles
                    psi_vec = psi_angles[:, 0] if psi_angles.ndim == 2 else psi_angles
                    # Flatten to 1D python floats
                    cv1_data.extend([float(v) for v in np.array(phi_vec).reshape(-1)])
                    cv2_data.extend([float(v) for v in np.array(psi_vec).reshape(-1)])
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
