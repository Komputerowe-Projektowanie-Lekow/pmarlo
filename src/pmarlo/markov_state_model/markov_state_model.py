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

from __future__ import annotations

import csv
import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, cast

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from scipy import constants
from scipy.ndimage import gaussian_filter
from scipy.sparse import csc_matrix, issparse, save_npz
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans

from pmarlo.states.msm_bridge import _row_normalize, _stationary_from_T
from pmarlo.utils.msm_utils import ensure_connected_counts

from ..cluster.micro import ClusteringResult, cluster_microstates
from ..replica_exchange.demux_metadata import DemuxMetadata
from ..results import FESResult, ITSResult, MSMResult
from .utils import safe_timescales

logger = logging.getLogger("pmarlo")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class CKTestResult:
    """Results of a Chapman–Kolmogorov test.

    Attributes:
        mse: Mapping from lag multiple to mean squared error.
        mode: Level at which the CK test was performed ("micro" or "macro").
        insufficient_data: Flag indicating if the test could not be evaluated due to
            limited statistics.
        thresholds: Threshold values used to determine data sufficiency.
    """

    mse: Dict[int, float] = field(default_factory=dict)
    mode: str = "micro"
    insufficient_data: bool = False
    thresholds: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the result."""

        return {
            "mse": {int(k): float(v) for k, v in self.mse.items()},
            "mode": self.mode,
            "insufficient_data": self.insufficient_data,
            "thresholds": self.thresholds,
        }


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
        random_state: int | None = 42,
    ):
        """
        Initialize the Enhanced MSM analyzer.

        Args:
            trajectory_files: Single trajectory file or list of files
            topology_file: Topology file (PDB) for the system
            temperatures: List of temperatures for TRAM analysis
            output_dir: Directory for output files
            random_state: Seed for internal stochastic components. ``None``
                uses the global random state.
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
        self.random_state = int(random_state) if random_state is not None else None

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

        # Explicit feature processing settings
        self.feature_stride: int = 1
        self.tica_lag: int = 0
        self.tica_components: Optional[int] = None
        self.effective_frames: int = 0
        self.raw_frames: int = 0

        self.time_per_frame_ps: Optional[float] = None
        self.demux_metadata: Optional[DemuxMetadata] = None
        self.total_frames: Optional[int] = None

        # Estimation controls
        self.estimator_backend: str = "deeptime"  # or "pmarlo" for fallback/debug
        self.count_mode: str = "sliding"  # "sliding" or "strided" counting

        # TRAM data
        self.tram_weights: Optional[np.ndarray] = None
        self.multi_temp_counts: Dict[float, Dict[Tuple[int, int], float]] = (
            {}
        )  # Fix: Added proper type annotation

        # Analysis results
        self.implied_timescales: Optional[ITSResult] = None
        self.state_table: Optional[pd.DataFrame] = (
            None  # Fix: Will be DataFrame when created
        )
        self.fes_data: Optional[Dict[str, Any]] = None

        logger.info(
            f"Enhanced MSM initialized for {len(self.trajectory_files)} trajectories"
        )

    def load_trajectories(
        self,
        stride: int = 1,
        atom_selection: str | Sequence[int] | None = None,
        chunk_size: int = 1000,
    ) -> None:
        """Load trajectory data for analysis.

        Trajectories are streamed from disk using :func:`mdtraj.iterload` to
        avoid loading the entire file into memory at once.  This is
        particularly useful for large DCD files.  When ``atom_selection`` is
        provided, only the selected atoms are loaded.

        Parameters
        ----------
        stride:
            Stride for loading frames (``1`` loads every frame).
        atom_selection:
            Either an MDTraj atom selection string or an explicit sequence of
            atom indices to retain.  ``None`` selects all atoms.
        chunk_size:
            Number of frames to read per chunk when streaming from disk.
        """

        logger.info("Loading trajectory data (streaming mode)...")

        atom_indices: Sequence[int] | None = None
        if atom_selection is not None:
            topo = md.load_topology(self.topology_file)
            if isinstance(atom_selection, str):
                atom_indices = topo.select(atom_selection)
            else:
                atom_indices = list(atom_selection)

        self.trajectories = []
        for i, traj_file in enumerate(self.trajectory_files):
            path = Path(traj_file)
            if not path.exists():
                logger.warning(f"Trajectory file not found: {traj_file}")
                continue

            logger.info(
                "Streaming trajectory %s with stride=%d, chunk=%d%s",
                traj_file,
                stride,
                chunk_size,
                f", selection={atom_selection}" if atom_selection else "",
            )
            joined: md.Trajectory | None = None
            from pmarlo.io import trajectory as traj_io

            for chunk in traj_io.iterload(
                traj_file,
                top=self.topology_file,
                stride=stride,
                atom_indices=atom_indices,
                chunk=chunk_size,
            ):
                joined = chunk if joined is None else joined.join(chunk)
            if joined is None:
                logger.warning(f"No frames loaded from {traj_file}")
                continue
            self.trajectories.append(joined)
            logger.info("Loaded trajectory %d: %d frames", i + 1, joined.n_frames)

            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists() and self.demux_metadata is None:
                try:
                    meta = DemuxMetadata.from_json(meta_path)
                    self.demux_metadata = meta
                    stride_frames = (
                        meta.exchange_frequency_steps // meta.frames_per_segment
                    )
                    self.frame_stride = stride_frames
                    self.time_per_frame_ps = (
                        meta.integration_timestep_ps * stride_frames
                    )
                    logger.info(
                        "Loaded demux metadata: stride=%d, dt=%.4f ps",
                        stride_frames,
                        self.time_per_frame_ps,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(f"Failed to parse metadata {meta_path}: {exc}")

        if not self.trajectories:
            raise ValueError("No trajectories loaded successfully")

        logger.info(f"Total trajectories loaded: {len(self.trajectories)}")
        try:
            self.total_frames = int(sum(int(t.n_frames) for t in self.trajectories))
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
        self,
        feature_type: str = "phi_psi",
        n_features: Optional[int] = None,
        feature_stride: int = 1,
        tica_lag: int = 0,
        tica_components: Optional[int] = None,
    ) -> None:
        """Compute features from trajectory data.

        Parameters
        ----------
        feature_type:
            Type of features to compute (``"phi_psi"``, ``"distances"``, ``"contacts"``).
        n_features:
            Number of features to compute (auto if ``None``).
        feature_stride:
            Subsampling stride applied to the trajectories before feature
            computation.
        tica_lag:
            Lag time (in frames) used for the optional TICA projection.  When
            greater than zero, the first ``tica_lag`` frames are discarded from
            each trajectory after projection to ensure consistent effective
            lengths.
        tica_components:
            Number of TICA components to keep.  When ``None`` and ``tica_lag`` is
            non-zero, the ``n_features`` hint is used.
        """
        self._log_compute_features_start(feature_type)
        self.feature_stride = int(max(1, feature_stride))
        self.tica_lag = int(max(0, tica_lag))
        self.tica_components = tica_components
        self.raw_frames = sum(traj.n_frames for traj in self.trajectories)

        proc_trajs = [traj[:: self.feature_stride] for traj in self.trajectories]
        self.trajectories = proc_trajs
        strided_frames = sum(traj.n_frames for traj in proc_trajs)

        all_features: List[np.ndarray] = []
        for traj in proc_trajs:
            traj_features = self._compute_features_for_traj(
                traj, feature_type, n_features
            )
            all_features.append(traj_features)
        self.features = self._combine_all_features(all_features)

        # Optional: apply TICA projection when requested
        if (
            tica_components is not None
            or self.tica_lag > 0
            or "tica" in feature_type.lower()
        ):
            self._maybe_apply_tica(tica_components or n_features, self.tica_lag)

        effective_frames = self.features.shape[0] if self.features is not None else 0
        self.effective_frames = effective_frames
        self._log_features_shape(self.raw_frames, strided_frames, effective_frames)

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
        """Backward-compatible φ/ψ features via new features API.

        Uses api.compute_features(["phi_psi"]) to get dihedral angles (radians)
        and returns trig-expanded features [cos(angles), sin(angles)] as before.
        Falls back to Cartesian if no angles found.
        """
        try:
            # Lazy import to avoid import cycles
            from pmarlo import api  # type: ignore

            X, _cols, _periodic = api.compute_features(traj, feature_specs=["phi_psi"])
            if X.size == 0 or X.shape[1] == 0:
                logger.warning("No dihedral angles found, using Cartesian coordinates")
                return self._fallback_cartesian_features(traj)
            # Apply trig expansion over all angle columns to preserve previous behavior
            return np.hstack([np.cos(X), np.sin(X)])
        except Exception:
            # Robust fallback to original direct computation path
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

    def _log_features_shape(
        self, raw_frames: int, strided_frames: int, effective_frames: int
    ) -> None:
        """Log frame accounting after feature computation."""
        if self.features is None:
            return
        logger.info(f"Features computed: ({strided_frames}, {self.features.shape[1]})")
        logger.info(
            "Raw frames: %d → %d after feature_stride=%d; effective frames after lag %d: %d",
            raw_frames,
            strided_frames,
            self.feature_stride,
            self.tica_lag,
            effective_frames,
        )

    def cluster_features(
        self,
        n_states: int | Literal["auto"] = "auto",
        algorithm: str = "kmeans",
        random_state: int | None = None,
    ) -> None:
        """Cluster features to create discrete states."""

        if self.features is None:
            raise ValueError("Features must be computed before clustering")

        rng = self.random_state if random_state is None else random_state
        logger.info(
            "Clustering features using %s: requested=%s",
            algorithm,
            n_states,
        )

        method_choice = cast(
            Literal["kmeans", "minibatchkmeans"],
            algorithm if algorithm in ["kmeans", "minibatchkmeans"] else "kmeans",
        )
        result: ClusteringResult = cluster_microstates(
            self.features,
            method=method_choice,
            n_states=n_states,
            random_state=rng,
        )
        labels = result.labels
        self.cluster_centers = result.centers

        # Split labels back into trajectories
        self.dtrajs = []
        start_idx = 0
        for traj in self.trajectories:
            end_idx = start_idx + traj.n_frames
            self.dtrajs.append(labels[start_idx:end_idx])
            start_idx = end_idx

        self.n_states = int(result.n_states)
        logger.info(
            "Clustering completed: requested=%s, actual=%d%s",
            n_states,
            self.n_states,
            f" ({result.rationale})" if result.rationale else "",
        )

    def build_msm(self, lag_time: int = 20, method: str = "standard") -> None:
        """Build Markov State Model from discrete trajectories.

        Parameters
        ----------
        lag_time:
            Lag time (in frames) used when counting transitions.  Values smaller
            than one are treated as one to avoid degenerate estimates.
        method:
            MSM construction method.  ``"standard"`` uses a discrete-time MSM
            and ``"tram"`` uses the TRAM estimator when multiple thermodynamic
            ensembles are available.
        """
        lag_time = int(max(1, lag_time))
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
                    self._maybe_apply_tica(3, self.tica_lag or self.lag_time)
        except Exception:
            pass

        if method == "standard":
            self._build_standard_msm(lag_time, count_mode=self.count_mode)
        elif method == "tram":
            self._build_tram_msm(lag_time)
        else:
            raise ValueError(f"Unknown MSM method: {method}")

        # Compute free energies
        self._compute_free_energies()

        logger.info("MSM construction completed")

    # ---------------- TICA (time-lagged ICA) support ----------------

    def _maybe_apply_tica(self, n_components_hint: Optional[int], lag: int) -> None:
        """Apply TICA via deeptime when requested.

        Parameters
        ----------
        n_components_hint:
            Desired number of TICA components.  ``None`` skips projection.
        lag:
            Lag time (in frames) used in the TICA estimation and to discard the
            first ``lag`` frames of each trajectory after transformation.
        """
        if self.features is None or n_components_hint is None:
            return
        n_components = int(max(2, min(5, n_components_hint)))
        try:
            from deeptime.decomposition import TICA as _DT_TICA  # type: ignore

            Xs: List[np.ndarray] = []
            start = 0
            for traj in self.trajectories:
                end = start + traj.n_frames
                Xs.append(self.features[start:end])
                start = end

            tica = _DT_TICA(lagtime=int(max(1, lag or 1)), dim=n_components)
            tica_model = tica.fit(Xs).fetch_model()
            Ys = [tica_model.transform(x)[int(max(0, lag)) :] for x in Xs]
            self.features = np.vstack(Ys)
            self.trajectories = [traj[int(max(0, lag)) :] for traj in self.trajectories]
            if hasattr(tica_model, "eigenvectors_"):
                self.tica_components_ = tica_model.eigenvectors_  # type: ignore[attr-defined]
            if hasattr(tica_model, "eigenvalues_"):
                self.tica_eigenvalues_ = tica_model.eigenvalues_  # type: ignore[attr-defined]
            self.tica_components = n_components
            logger.info("Applied deeptime TICA to %d components", n_components)
        except Exception as e:
            logger.warning("deeptime TICA failed (%s); proceeding without TICA", e)
            if lag > 0:
                self.features = self.features[int(max(0, lag)) :]
                self.trajectories = [
                    traj[int(max(0, lag)) :] for traj in self.trajectories
                ]

    def _build_standard_msm(self, lag_time: int, count_mode: str = "sliding") -> None:
        """Estimate transition counts and matrix for a discrete MSM.

        Parameters
        ----------
        lag_time:
            Lag time (in frames) for the transition counts.
        count_mode:
            ``"sliding"`` counts transitions between every consecutive frame
            separated by ``lag_time`` while ``"strided"`` advances the starting
            frame by ``lag_time`` after each counted transition.
        """

        if self.effective_frames and lag_time >= self.effective_frames:
            raise ValueError(
                f"lag_time {lag_time} exceeds available effective frames {self.effective_frames}"
            )

        use_deeptime = False
        if self.estimator_backend == "deeptime":
            try:  # pragma: no cover - exercised in environments with deeptime
                from deeptime.markov import TransitionCountEstimator  # type: ignore

                use_deeptime = True
            except Exception:  # pragma: no cover - import failure
                use_deeptime = False

        lag = int(max(1, lag_time))
        max_valid_lag = min(len(dt) for dt in self.dtrajs) - 1 if self.dtrajs else 0
        if lag > max_valid_lag and max_valid_lag > 0:
            logger.warning(
                "Lag %s exceeds max feasible %s; capping", lag, max_valid_lag
            )
            lag = max_valid_lag
        if max_valid_lag < 1:
            self.count_matrix = np.zeros((self.n_states, self.n_states), dtype=float)
            self.transition_matrix = np.eye(self.n_states, dtype=float)
            self.stationary_distribution = np.zeros((self.n_states,), dtype=float)
            return

        if use_deeptime:
            tce = TransitionCountEstimator(
                lagtime=lag,
                count_mode="sliding" if count_mode == "strided" else str(count_mode),
                sparse=False,
            )
            count_model = tce.fit(self.dtrajs).fetch_model()
            counts = np.asarray(count_model.count_matrix, dtype=float)
        else:
            counts = np.zeros((self.n_states, self.n_states), dtype=float)
            step = lag if count_mode == "strided" else 1
            for dtraj in self.dtrajs:
                if len(dtraj) <= lag:
                    continue
                for i in range(0, len(dtraj) - lag, step):
                    state_i = int(dtraj[i])
                    state_j = int(dtraj[i + lag])
                    if state_i < 0 or state_j < 0:
                        continue
                    if state_i >= self.n_states or state_j >= self.n_states:
                        continue
                    counts[state_i, state_j] += 1.0

        res = ensure_connected_counts(counts)
        self.count_matrix = np.zeros((self.n_states, self.n_states), dtype=float)
        if res.counts.size:
            self.count_matrix[np.ix_(res.active, res.active)] = res.counts
            T_active = _row_normalize(res.counts)
            pi_active = _stationary_from_T(T_active)
            T_full = np.eye(self.n_states, dtype=float)
            T_full[np.ix_(res.active, res.active)] = T_active
            pi_full = np.zeros((self.n_states,), dtype=float)
            pi_full[res.active] = pi_active
        else:
            T_full = np.eye(self.n_states, dtype=float)
            pi_full = np.zeros((self.n_states,), dtype=float)

        self.transition_matrix = T_full
        self.stationary_distribution = pi_full

    def _build_tram_msm(self, lag_time: int):
        """Build MSM using deeptime TRAM for multi-ensemble data when available."""
        logger.info("Building TRAM MSM for multi-temperature data via deeptime...")

        if len(self.temperatures) <= 1:
            logger.warning("Only one ensemble provided, falling back to standard MSM")
            return self._build_standard_msm(lag_time)

        # Expect dtrajs to be a list aligned with ensembles; TRAM also needs bias/energies—
        # assume user supplies them via attributes if available.
        try:
            from deeptime.markov.msm import TRAM, TRAMDataset  # type: ignore

            # Build dataset: minimal path uses only dtrajs with equal weights when
            # bias/energies are not provided; recommend users to set self.bias_matrices.
            # If bias is unavailable, the estimator may not converge—guard accordingly.
            bias = getattr(self, "bias_matrices", None)
            if bias is None:
                logger.warning(
                    "No bias matrices provided for TRAM; please set self.bias_matrices. Falling back to standard MSM."
                )
                return self._build_standard_msm(lag_time)

            ds = TRAMDataset(  # type: ignore[call-arg]
                dtrajs=self.dtrajs,
                bias_matrices=bias,
            )
            tram = TRAM(
                lagtime=int(max(1, lag_time)),
                count_mode="sliding",
                init_strategy="MBAR",
            )
            tram_model = tram.fit(ds).fetch_model()

            # Extract per-ensemble MSMs and counts; choose reference
            ref = int(getattr(self, "tram_reference_index", 0))
            msms = getattr(tram_model, "msms", None)
            cm_list = getattr(tram_model, "count_models", None)
            if isinstance(msms, list) and 0 <= ref < len(msms):
                msm_ref = msms[ref]
                self.transition_matrix = np.asarray(
                    msm_ref.transition_matrix, dtype=float
                )
                if (
                    hasattr(msm_ref, "stationary_distribution")
                    and msm_ref.stationary_distribution is not None
                ):
                    self.stationary_distribution = np.asarray(
                        msm_ref.stationary_distribution, dtype=float
                    )
                if isinstance(cm_list, list) and 0 <= ref < len(cm_list):
                    self.count_matrix = np.asarray(
                        cm_list[ref].count_matrix, dtype=float
                    )
            else:
                logger.warning(
                    "TRAM did not expose per-ensemble MSMs; falling back to standard MSM"
                )
                return self._build_standard_msm(lag_time)
        except Exception as e:
            logger.warning(
                f"deeptime TRAM unavailable or failed ({e}); using standard MSM"
            )
            return self._build_standard_msm(lag_time)

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
        self,
        lag_times: Optional[List[int]] = None,
        n_timescales: int = 5,
        *,
        n_samples: int = 100,
        ci: float = 0.95,
        dirichlet_alpha: float = 1e-3,
        plateau_m: int | None = None,
        plateau_epsilon: float = 0.1,
    ) -> None:
        """Estimate implied timescales with Bayesian uncertainty.

        Parameters
        ----------
        lag_times:
            List of lag times to test. ``None`` uses a default ladder.
        n_timescales:
            Number of timescales to compute.
        n_samples:
            Number of transition matrix samples drawn from the posterior.
        ci:
            Confidence interval level between 0 and 1.
        dirichlet_alpha:
            Dirichlet pseudocount added to transition counts.
        plateau_m:
            Number of leading eigenvalues considered for plateau detection.
        plateau_epsilon:
            Maximum relative change tolerated between consecutive eigenvalues
            when selecting a plateau window.
        """

        logger.info("Computing implied timescales with Bayesian estimation")

        lag_times = self._its_default_lag_times(lag_times)

        if not getattr(self, "dtrajs", None):
            logger.warning("No trajectories available for implied timescales")
            empty = ITSResult(
                lag_times=np.array([], dtype=int),
                eigenvalues=np.empty((0, n_timescales)),
                eigenvalues_ci=np.empty((0, n_timescales, 2)),
                timescales=np.empty((0, n_timescales)),
                timescales_ci=np.empty((0, n_timescales, 2)),
                rates=np.empty((0, n_timescales)),
                rates_ci=np.empty((0, n_timescales, 2)),
            )
            self.implied_timescales = empty
            return

        max_valid_lag = min(len(dt) for dt in self.dtrajs) - 1
        if max_valid_lag < 1:
            logger.warning("Trajectories too short for implied timescales")
            empty = ITSResult(
                lag_times=np.array([], dtype=int),
                eigenvalues=np.empty((0, n_timescales)),
                eigenvalues_ci=np.empty((0, n_timescales, 2)),
                timescales=np.empty((0, n_timescales)),
                timescales_ci=np.empty((0, n_timescales, 2)),
                rates=np.empty((0, n_timescales)),
                rates_ci=np.empty((0, n_timescales, 2)),
            )
            self.implied_timescales = empty
            return

        original_lag_times = list(lag_times)
        if any(lag_val > max_valid_lag for lag_val in original_lag_times):
            logger.warning("Capping lag times above max_valid_lag=%s", max_valid_lag)
        lag_times = [lt for lt in original_lag_times if 1 <= lt <= max_valid_lag]
        if not lag_times:
            logger.warning("No valid lag times after capping")
            empty = ITSResult(
                lag_times=np.array([], dtype=int),
                eigenvalues=np.empty((0, n_timescales)),
                eigenvalues_ci=np.empty((0, n_timescales, 2)),
                timescales=np.empty((0, n_timescales)),
                timescales_ci=np.empty((0, n_timescales, 2)),
                rates=np.empty((0, n_timescales)),
                rates_ci=np.empty((0, n_timescales, 2)),
            )
            self.implied_timescales = empty
            return

        max_lag = max(lag_times)
        eff = getattr(self, "effective_frames", None)
        if eff is not None and eff > 0 and max_lag >= eff:
            raise ValueError(
                f"Maximum lag {max_lag} exceeds available effective frames {eff}"
            )

        if self.random_state is not None:
            np.random.seed(self.random_state)
            try:
                import random as _random

                _random.seed(self.random_state)
            except Exception:
                pass

        eval_means: list[list[float]] = []
        eval_ci: list[list[list[float]]] = []
        ts_means: list[list[float]] = []
        ts_ci: list[list[list[float]]] = []
        rate_means: list[list[float]] = []
        rate_ci: list[list[list[float]]] = []

        q_low = 50.0 * (1.0 - ci)
        q_high = 100.0 - q_low

        for lag in lag_times:
            try:
                from deeptime.markov import TransitionCountEstimator  # type: ignore
                from deeptime.markov.msm import (  # type: ignore
                    BayesianMSM,
                    MaximumLikelihoodMSM,
                )
            except Exception as e:  # pragma: no cover
                logger.error("deeptime import failed: %s", e)
                raise

            tce = TransitionCountEstimator(
                lagtime=int(max(1, lag)), count_mode=self.count_mode, sparse=False
            )
            C = np.asarray(tce.fit(self.dtrajs).fetch_model().count_matrix, dtype=float)
            res = ensure_connected_counts(C, alpha=dirichlet_alpha)
            if res.counts.size == 0:
                eval_means.append([np.nan] * n_timescales)
                eval_ci.append([[np.nan, np.nan]] * n_timescales)
                ts_means.append([np.nan] * n_timescales)
                ts_ci.append([[np.nan, np.nan]] * n_timescales)
                rate_means.append([np.nan] * n_timescales)
                rate_ci.append([[np.nan, np.nan]] * n_timescales)
                continue

            matrices: np.ndarray
            try:
                bmsm = BayesianMSM(n_samples=n_samples, reversible=True)
                posterior = bmsm.fit_from_counts(res.counts).fetch_model()
                matrices = np.array(
                    [m.transition_matrix for m in posterior.samples], dtype=float
                )
            except Exception as e:
                logger.warning(
                    "Bayesian MSM failed for lag %s: %s; using ML estimator", lag, e
                )
                try:
                    ml = MaximumLikelihoodMSM(reversible=True)
                    model = ml.fit_from_counts(res.counts).fetch_model()
                    matrices = np.expand_dims(model.transition_matrix, 0)
                except Exception as e2:  # pragma: no cover
                    logger.error(
                        "Maximum likelihood MSM failed for lag %s: %s", lag, e2
                    )
                    matrices = np.empty((0, res.counts.shape[0], res.counts.shape[1]))

            if matrices.size == 0:
                eval_means.append([np.nan] * n_timescales)
                eval_ci.append([[np.nan, np.nan]] * n_timescales)
                ts_means.append([np.nan] * n_timescales)
                ts_ci.append([[np.nan, np.nan]] * n_timescales)
                rate_means.append([np.nan] * n_timescales)
                rate_ci.append([[np.nan, np.nan]] * n_timescales)
                continue

            eig_samples: list[np.ndarray] = []
            for T in matrices:
                eigenvals = np.real(np.linalg.eigvals(T))
                eigenvals = np.sort(eigenvals)[::-1]
                eig_samples.append(eigenvals[1 : n_timescales + 1])
            eig_arr = np.asarray(eig_samples, dtype=float)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

                eval_mean = np.nanmean(eig_arr, axis=0)
                eval_lo = np.nanpercentile(eig_arr, q_low, axis=0)
                eval_hi = np.nanpercentile(eig_arr, q_high, axis=0)

                ts_arr = safe_timescales(lag, eig_arr)
                rate_arr = np.reciprocal(
                    ts_arr, where=np.isfinite(ts_arr), out=np.full_like(ts_arr, np.nan)
                )

                ts_mean = np.nanmean(ts_arr, axis=0)
                ts_lo = np.nanpercentile(ts_arr, q_low, axis=0)
                ts_hi = np.nanpercentile(ts_arr, q_high, axis=0)

                rate_mean = np.nanmean(rate_arr, axis=0)
                rate_lo = np.nanpercentile(rate_arr, q_low, axis=0)
                rate_hi = np.nanpercentile(rate_arr, q_high, axis=0)

            if np.all(rate_mean < 1e-12):
                logger.warning(
                    "Rates collapsed to near zero at lag %s; data may be insufficient",
                    lag,
                )

            eval_means.append(eval_mean.tolist())
            eval_ci.append(np.stack([eval_lo, eval_hi], axis=1).tolist())
            ts_means.append(ts_mean.tolist())
            ts_ci.append(np.stack([ts_lo, ts_hi], axis=1).tolist())
            rate_means.append(rate_mean.tolist())
            rate_ci.append(np.stack([rate_lo, rate_hi], axis=1).tolist())

        eigen_means_arr = np.asarray(eval_means)
        eigen_ci_arr = np.asarray(eval_ci)
        ts_means_arr = np.asarray(ts_means)
        ts_ci_arr = np.asarray(ts_ci)
        rate_means_arr = np.asarray(rate_means)
        rate_ci_arr = np.asarray(rate_ci)

        plateau_m = plateau_m or n_timescales
        recommended = self._select_lag_window(
            np.array(lag_times), eigen_means_arr, plateau_m, plateau_epsilon
        )

        self.implied_timescales = ITSResult(
            lag_times=np.array(lag_times, dtype=int),
            eigenvalues=eigen_means_arr,
            eigenvalues_ci=eigen_ci_arr,
            timescales=ts_means_arr,
            timescales_ci=ts_ci_arr,
            rates=rate_means_arr,
            rates_ci=rate_ci_arr,
            recommended_lag_window=recommended,
        )

        logger.info("Implied timescales computation completed")

    def _its_default_lag_times(self, lag_times: Optional[List[int]]) -> List[int]:
        if lag_times is None:
            return [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200]
        return [int(max(1, v)) for v in lag_times]

    def _select_lag_window(
        self,
        lag_times: np.ndarray,
        eigenvalues: np.ndarray,
        m: int,
        epsilon: float,
    ) -> Optional[tuple[int, int]]:
        """Return lag window where first ``m`` eigenvalues vary ≤ ``epsilon``."""

        if lag_times.size < 2 or eigenvalues.size == 0:
            return None

        stable = []
        for i in range(len(lag_times) - 1):
            rel = np.abs(eigenvalues[i + 1, :m] - eigenvalues[i, :m]) / np.maximum(
                np.abs(eigenvalues[i, :m]), 1e-12
            )
            stable.append(bool(np.all(rel <= epsilon)))

        longest: Optional[tuple[int, int]] = None
        i = 0
        while i < len(stable):
            if stable[i]:
                j = i
                while j < len(stable) and stable[j]:
                    j += 1
                if longest is None or (j - i) > (longest[1] - longest[0]):
                    longest = (i, j)
                i = j
            else:
                i += 1

        if longest and (longest[1] - longest[0]) >= 1:
            return int(lag_times[longest[0]]), int(lag_times[longest[1]])
        return None

    # ---------------- TPT / MFPT / Committors (deeptime-backed) ----------------

    def mfpt(self, A: List[int], B: List[int]) -> Optional[float]:
        """Mean first passage time between sets of microstates using deeptime MSM.

        Returns MFPT in units of lag steps; converts to frames implicitly.
        """
        try:
            from deeptime.markov import TransitionCountEstimator  # type: ignore
            from deeptime.markov.msm import MaximumLikelihoodMSM  # type: ignore

            tce = TransitionCountEstimator(
                lagtime=int(max(1, self.lag_time)), count_mode="sliding", sparse=False
            )
            count_model = tce.fit(self.dtrajs).fetch_model()
            ml = MaximumLikelihoodMSM(reversible=True)
            msm = ml.fit(count_model).fetch_model()
            return float(msm.mfpt(A, B))  # type: ignore[attr-defined]
        except Exception:
            return None

    def committor(self, A: List[int], B: List[int]) -> Optional[np.ndarray]:
        """Compute forward committor q+ for sets A and B using deeptime if available."""
        try:
            from deeptime.markov import TransitionCountEstimator  # type: ignore
            from deeptime.markov.msm import MaximumLikelihoodMSM  # type: ignore

            tce = TransitionCountEstimator(
                lagtime=int(max(1, self.lag_time)), count_mode="sliding", sparse=False
            )
            count_model = tce.fit(self.dtrajs).fetch_model()
            ml = MaximumLikelihoodMSM(reversible=True)
            msm = ml.fit(count_model).fetch_model()
            return np.asarray(msm.committor(A, B), dtype=float)  # type: ignore[attr-defined]
        except Exception:
            return None

    def tpt_flux(self, A: List[int], B: List[int]) -> Optional[Dict[str, Any]]:
        """Reactive flux network between A and B using deeptime TPT tools if available."""
        try:
            from deeptime.markov import TransitionCountEstimator  # type: ignore
            from deeptime.markov.msm import MaximumLikelihoodMSM  # type: ignore

            # Some versions expose TPT via msm.tpt(A,B)
            tce = TransitionCountEstimator(
                lagtime=int(max(1, self.lag_time)), count_mode="sliding", sparse=False
            )
            count_model = tce.fit(self.dtrajs).fetch_model()
            ml = MaximumLikelihoodMSM(reversible=True)
            msm = ml.fit(count_model).fetch_model()
            try:
                tpt_obj = msm.tpt(A, B)  # type: ignore[attr-defined]
                # Extract minimal useful fields if available
                out: Dict[str, Any] = {}
                for key in (
                    "flux",
                    "stationary_distribution",
                    "committor_forward",
                    "committor_backward",
                ):
                    if hasattr(tpt_obj, key):
                        out[key] = getattr(tpt_obj, key)
                return out if out else {"tpt": tpt_obj}
            except Exception:
                return None
        except Exception:
            return None

    # ---------------- CK test via internal microstate counts ----------------

    def compute_ck_test_micro(
        self,
        factors: Optional[List[int]] = None,
        max_states: int = 50,
        min_transitions: int = 5,
    ) -> CKTestResult:
        """Compute CK test MSE at the microstate level.

        The calculation is restricted to the largest connected component of the
        transition graph, limited to ``max_states`` states. If any state has fewer
        than ``min_transitions`` outgoing transitions, the result is marked as
        insufficient.
        """

        factors = self._normalize_ck_factors(factors)
        result = CKTestResult(
            mode="micro",
            thresholds={
                "min_transitions_per_state": int(min_transitions),
                "max_states": int(max_states),
            },
        )
        if not self.dtrajs or self.n_states <= 1 or self.lag_time <= 0:
            result.insufficient_data = True
            return result

        T_all, C_all = self._count_micro_T(
            self.dtrajs, self.n_states, int(self.lag_time)
        )
        idx = self._largest_connected_states(C_all, int(max_states))
        if idx.size == 0:
            result.insufficient_data = True
            return result

        state_map = {int(old): i for i, old in enumerate(idx)}
        filtered = [
            np.array([state_map[s] for s in traj if s in state_map], dtype=int)
            for traj in self.dtrajs
        ]
        n_sel = int(len(idx))
        T1, C1 = self._count_micro_T(filtered, n_sel, int(self.lag_time))
        if np.any(C1.sum(axis=1) < min_transitions):
            result.insufficient_data = True
            return result

        for f in factors:
            T_emp, Ck = self._count_micro_T(
                filtered, n_sel, int(self.lag_time) * int(f)
            )
            if np.any(Ck.sum(axis=1) < min_transitions):
                result.insufficient_data = True
                return result
            T_theory = np.linalg.matrix_power(T1, int(f))
            diff = T_theory - T_emp
            result.mse[int(f)] = float(np.mean(diff * diff))

        self._persist_ck_micro_results(result, n_sel, int(self.lag_time))
        return result

    def select_lag_time_ck(
        self,
        tau_candidates: Sequence[int],
        factor: int = 2,
        mse_epsilon: float = 0.05,
    ) -> int:
        """Select lag time by CK MSE plateau with monotonic slowest ITS.

        Parameters
        ----------
        tau_candidates:
            Iterable of lag times (in frames) to evaluate.
        factor:
            Lag-time multiple for the CK test (default is ``2``).
        mse_epsilon:
            Relative MSE improvement threshold to declare a plateau.

        Returns
        -------
        int
            Selected lag time (frames).
        """

        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        mses: list[float] = []
        taus: list[int] = []
        prev_mse: float | None = None
        prev_its: float | None = None
        selected = int(tau_candidates[0])

        for tau in tau_candidates:
            tau = int(tau)
            T1, _ = self._count_micro_T(self.dtrajs, self.n_states, tau)
            # slowest implied timescale
            evals = np.sort(np.real(np.linalg.eigvals(T1)))[::-1]
            if len(evals) > 1 and 0 < evals[1] < 1:
                ts = -tau / np.log(evals[1])
            else:
                ts = float("inf")
            taus.append(tau)

            T_emp, Ck = self._count_micro_T(
                self.dtrajs, self.n_states, tau * int(factor)
            )
            if np.any(Ck.sum(axis=1) == 0):
                mse = float("inf")
            else:
                T_theory = np.linalg.matrix_power(T1, int(factor))
                diff = T_theory - T_emp
                mse = float(np.mean(diff * diff))
            mses.append(mse)

            if prev_its is not None and ts < prev_its:
                break
            if prev_mse is not None:
                if mse > prev_mse:
                    break
                rel = (prev_mse - mse) / max(prev_mse, 1e-12)
                if rel <= mse_epsilon:
                    break
            selected = tau
            prev_mse = mse
            prev_its = ts
        else:
            # no early break; pick minimal MSE
            idx = int(np.nanargmin(mses))
            selected = taus[idx]

        # persist CSV
        csv_path = out / "ck_mse.csv"
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tau", "mse"])
            for t, m in zip(taus, mses):
                writer.writerow([t, m])

        # plot
        plt.figure()
        if mses:
            plt.plot(taus, mses, marker="o")
            plt.xlabel("lag time (frames)")
            plt.ylabel("CK MSE")
        plt.tight_layout()
        try:
            plt.savefig(out / "ck.png")
        finally:
            plt.close()

        self.lag_time = int(selected)
        tau_ps = (
            float(selected) * float(self.time_per_frame_ps)
            if self.time_per_frame_ps is not None
            else float(selected)
        )
        print(f"Selected τ = {tau_ps} ps by CK")
        return int(selected)

    # ---------------- Macrostate CK test with eigen-gap check ----------------

    def _micro_to_macro_labels(self, n_macrostates: int = 3) -> Optional[np.ndarray]:
        try:
            if (
                self.state_table is not None
                and "macrostate" in self.state_table.columns
            ):
                labels = np.asarray(self.state_table["macrostate"], dtype=int)
                if labels.size == self.n_states:
                    return labels
        except Exception:
            pass
        return self._pcca_lumping(n_macrostates=n_macrostates)

    def compute_ck_test_macrostates(
        self,
        n_macrostates: int = 3,
        factors: Optional[List[int]] = None,
        min_transitions: int = 5,
    ) -> CKTestResult:
        """Compute CK test MSE at the macrostate level.

        Falls back to a microstate CK test if the eigenvalue gap of the underlying
        microstate transition matrix does not support at least two metastable
        states.
        """

        factors = self._normalize_ck_factors(factors)
        gap = self._micro_eigen_gap(k=2)
        if gap is None or gap <= 0.01:
            logger.info("Eigen gap too small; using microstate CK test instead")
            return self.compute_ck_test_micro(factors=factors)

        result = CKTestResult(
            mode="macro", thresholds={"min_transitions_per_state": int(min_transitions)}
        )
        if not self.dtrajs or self.n_states <= 0 or self.lag_time <= 0:
            result.insufficient_data = True
            return result

        macro_labels = self._micro_to_macro_labels(n_macrostates=n_macrostates)
        if macro_labels is None:
            return self.compute_ck_test_micro(factors=factors)
        n_macros = int(np.max(macro_labels) + 1)
        if n_macros <= 1:
            return self.compute_ck_test_micro(factors=factors)

        macro_trajs = self._build_macro_trajectories(self.dtrajs, macro_labels)
        T1, C1 = self._count_macro_T_and_counts(
            macro_trajs, n_macros, int(self.lag_time)
        )
        if np.any(C1.sum(axis=1) < min_transitions):
            result.insufficient_data = True
            return result

        for f in factors:
            mse, Ck = self._ck_mse_for_factor(
                T1, macro_trajs, n_macros, int(self.lag_time), int(f)
            )
            if mse is None or np.any(Ck.sum(axis=1) < min_transitions):
                result.insufficient_data = True
                return result
            result.mse[int(f)] = float(mse)

        self._persist_ck_macro_results(result, n_macros, int(self.lag_time))
        return result

    # ---- CK helpers ----

    def _normalize_ck_factors(self, factors: Optional[List[int]]) -> List[int]:
        if factors is None:
            return [2, 3, 4, 5]
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

    def _normalize_counts(self, C: np.ndarray) -> np.ndarray:
        rows = C.sum(axis=1)
        rows[rows == 0] = 1.0
        return C / rows[:, None]

    def _count_macro_T_and_counts(
        self, macro_trajs: List[np.ndarray], nM: int, lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        C = np.zeros((nM, nM), dtype=float)
        for seq in macro_trajs:
            if seq.size <= lag:
                continue
            for i in range(0, seq.size - lag):
                a = int(seq[i])
                b = int(seq[i + lag])
                if 0 <= a < nM and 0 <= b < nM:
                    C[a, b] += 1.0
        return self._normalize_counts(C), C

    def _count_macro_T(
        self, macro_trajs: List[np.ndarray], nM: int, lag: int
    ) -> np.ndarray:
        T, _ = self._count_macro_T_and_counts(macro_trajs, nM, lag)
        return T

    def _ck_mse_for_factor(
        self,
        T1: np.ndarray,
        macro_trajs: List[np.ndarray],
        nM: int,
        base_lag: int,
        factor: int,
    ) -> Tuple[Optional[float], np.ndarray]:
        try:
            T_theory = np.linalg.matrix_power(T1, int(factor))
            T_emp, C_emp = self._count_macro_T_and_counts(
                macro_trajs, nM, int(base_lag) * int(factor)
            )
            diff = T_theory - T_emp
            return float(np.mean(diff * diff)), C_emp
        except Exception:
            return None, np.zeros((nM, nM), dtype=float)

    def _count_micro_T(
        self, dtrajs: List[np.ndarray], nS: int, lag: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        C = np.zeros((nS, nS), dtype=float)
        for seq in dtrajs:
            seq = np.asarray(seq, dtype=int)
            if seq.size <= lag:
                continue
            for i in range(0, seq.size - lag):
                a = int(seq[i])
                b = int(seq[i + lag])
                if 0 <= a < nS and 0 <= b < nS:
                    C[a, b] += 1.0
        return self._normalize_counts(C), C

    def _largest_connected_states(self, C: np.ndarray, max_states: int) -> np.ndarray:
        try:
            adj = ((C + C.T) > 0).astype(int)
            _, labels = connected_components(adj, directed=False, return_labels=True)
            counts = np.bincount(labels)
            main = int(np.argmax(counts))
            idx = np.where(labels == main)[0]
            if idx.size > max_states:
                totals = (C + C.T).sum(axis=1)
                idx = idx[np.argsort(totals[idx])[::-1]][:max_states]
            return idx
        except Exception:
            return np.arange(min(C.shape[0], max_states))

    def _micro_eigen_gap(self, k: int = 2) -> Optional[float]:
        try:
            if self.transition_matrix is not None:
                T = np.asarray(self.transition_matrix, dtype=float)
            else:
                T, _ = self._count_micro_T(
                    self.dtrajs, self.n_states, int(self.lag_time)
                )
            evals = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
            if len(evals) <= k:
                return None
            return float(evals[k - 1] - evals[k])
        except Exception:
            return None

    def _persist_ck_macro_results(
        self, result: CKTestResult, nM: int, lag_frames: int
    ) -> None:
        try:
            out = {
                "n_macrostates": int(nM),
                "lag_time_frames": int(lag_frames),
                "factors": {str(k): v for k, v in result.mse.items()},
                "thresholds": result.thresholds,
                "mode": result.mode,
                "insufficient_data": result.insufficient_data,
            }
            with open(
                self.output_dir / "msm_analysis_ck_macro.json", "w", encoding="utf-8"
            ) as f:
                json.dump(out, f, indent=2)
        except Exception:
            pass

    def _persist_ck_micro_results(
        self, result: CKTestResult, nS: int, lag_frames: int
    ) -> None:
        try:
            out = {
                "n_states": int(nS),
                "lag_time_frames": int(lag_frames),
                "factors": {str(k): v for k, v in result.mse.items()},
                "thresholds": result.thresholds,
                "mode": result.mode,
                "insufficient_data": result.insufficient_data,
            }
            with open(
                self.output_dir / "msm_analysis_ck_micro.json", "w", encoding="utf-8"
            ) as f:
                json.dump(out, f, indent=2)
        except Exception:
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
        # Optional quick diagnostic: warn if φ and ψ look almost identical
        try:
            if cv1_name == "phi" and cv2_name == "psi" and len(cv1_data) > 10:
                corr = float(np.corrcoef(cv1_data, cv2_data)[0, 1])
                if abs(corr) > 0.95:
                    logger.warning(
                        (
                            "High linear correlation between ϕ and ψ (|r|=%.2f). "
                            "Check indexing/wrapping or data coverage."
                        ),
                        corr,
                    )
        except Exception:
            pass

        H, xedges, yedges = self._compute_weighted_histogram(
            cv1_data,
            cv2_data,
            frame_weights_array,
            bins,
            ranges,
            smooth_sigma=0.6,
            periodic=(cv1_name == "phi" and cv2_name == "psi"),
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
        """Compute per-frame weights using deeptime MSM when available; fallback to π mapping."""
        # Try to reconstruct a deeptime MSM model to leverage connected-set safe weights
        try:
            from deeptime.markov import TransitionCountEstimator  # type: ignore
            from deeptime.markov.msm import MaximumLikelihoodMSM  # type: ignore

            tce = TransitionCountEstimator(
                lagtime=int(max(1, self.lag_time)), count_mode="sliding", sparse=False
            )
            count_model = tce.fit(self.dtrajs).fetch_model()
            ml = MaximumLikelihoodMSM(reversible=True)
            msm = ml.fit(count_model).fetch_model()
            # compute_trajectory_weights returns list aligned with dtrajs
            weights_list = msm.compute_trajectory_weights(self.dtrajs)
            # Concatenate in the same order as frames
            return np.concatenate([np.asarray(w, dtype=float) for w in weights_list])
        except Exception:
            # Fallback: map stationary distribution onto frames by state labels
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
        periodic: bool = False,
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
                # Use circular smoothing on angular FES to respect periodic boundaries
                mode = "wrap" if periodic else "reflect"
                H = gaussian_filter(H, sigma=float(smooth_sigma), mode=mode)
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
        state_data: Dict[str, Any] = {"state_id": range(self.n_states)}
        frame_counts, total_frames = self._count_frames_per_state()
        state_data["counts"] = frame_counts.astype(int)
        population = frame_counts / max(total_frames, 1)
        state_data["population"] = population
        kT = (
            constants.k
            * float(self.temperatures[0])
            * constants.Avogadro
            / 1000.0
        )
        free_from_pop = -kT * np.log(np.clip(population, 1e-12, None))
        state_data["free_energy_kJ_mol"] = free_from_pop
        if self.free_energies is not None:
            diff = np.abs(free_from_pop - self.free_energies)
            if np.any(diff > 0.2):
                logger.warning(
                    "Free energies from populations differ from MSM output by more than 0.2 kJ/mol"
                )
        representative_frames, _ = self._find_representatives()
        rep_traj_array, rep_frame_array = self._representative_arrays(
            representative_frames
        )
        state_data["representative_traj"] = rep_traj_array
        state_data["representative_frame"] = rep_frame_array
        state_data["representative_frame_index"] = self._global_rep_indices(
            representative_frames
        )
        mean_phi_deg, mean_psi_deg = self._mean_phi_psi_per_state()
        state_data["mean_phi_deg"] = mean_phi_deg
        state_data["mean_psi_deg"] = mean_psi_deg
        fe_err = self._bootstrap_free_energy_errors(frame_counts)
        state_data["free_energy_error"] = fe_err
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
        return {"state_id": range(self.n_states)}

    def _count_frames_per_state(self) -> tuple[np.ndarray, int]:
        frame_counts = np.zeros(self.n_states)
        total_frames = 0
        for dtraj in self.dtrajs:
            for state in dtraj:
                frame_counts[state] += 1
                total_frames += 1
        return frame_counts, total_frames

    def _mean_phi_psi_per_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.features is None or self.features.shape[1] < 2:
            return (
                np.full(self.n_states, np.nan),
                np.full(self.n_states, np.nan),
            )
        phi_sum = np.zeros(self.n_states)
        psi_sum = np.zeros(self.n_states)
        counts = np.zeros(self.n_states)
        frame_idx = 0
        for dtraj in self.dtrajs:
            for state in dtraj:
                phi_sum[state] += self.features[frame_idx, 0]
                psi_sum[state] += self.features[frame_idx, 1]
                counts[state] += 1
                frame_idx += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_phi = np.rad2deg(phi_sum / counts)
            mean_psi = np.rad2deg(psi_sum / counts)
        return mean_phi, mean_psi

    def _global_rep_indices(
        self, representative_frames: list[tuple[int, int]]
    ) -> np.ndarray:
        lengths = [len(d) for d in self.dtrajs]
        offsets = np.cumsum([0] + lengths[:-1])
        indices: list[int] = []
        for traj_idx, local in representative_frames:
            if traj_idx < 0 or local < 0:
                indices.append(-1)
            else:
                indices.append(int(offsets[traj_idx] + local))
        return np.array(indices)

    def _bootstrap_counts(
        self, assignments: np.ndarray, n_boot: int = 200
    ) -> np.ndarray:
        rng = np.random.default_rng()
        samples = np.empty((n_boot, self.n_states), dtype=float)
        for i in range(n_boot):
            resample = rng.choice(assignments, size=assignments.size, replace=True)
            samples[i] = np.bincount(resample, minlength=self.n_states)
        return samples

    def _bootstrap_free_energy_errors(
        self, counts: np.ndarray, n_boot: int = 200
    ) -> np.ndarray:
        if not self.dtrajs:
            return np.zeros(self.n_states)
        assignments = np.concatenate(self.dtrajs)
        samples = self._bootstrap_counts(assignments, n_boot)
        kT = (
            constants.k
            * float(self.temperatures[0])
            * constants.Avogadro
            / 1000.0
        )
        fe_samples = -kT * np.log(np.clip(samples / assignments.size, 1e-12, None))
        return np.nanstd(fe_samples, axis=0)

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
        """PCCA+ metastable decomposition using deeptime when available; fallback to k-means spectral."""
        try:
            if self.transition_matrix is None or self.n_states <= n_macrostates:
                return None
            from deeptime.markov import pcca as _pcca  # type: ignore

            T = np.asarray(self.transition_matrix, dtype=float)
            model = _pcca(T, n_metastable_sets=int(n_macrostates))
            chi = np.asarray(model.memberships, dtype=float)
            labels = np.argmax(chi, axis=1)
            return labels.astype(int)
        except Exception:
            try:
                if self.transition_matrix is None or self.n_states <= n_macrostates:
                    return None
                T = np.asarray(self.transition_matrix, dtype=float)
                eigvals, eigvecs = np.linalg.eig(T.T)
                order = np.argsort(-np.real(eigvals))
                k = max(2, min(n_macrostates, T.shape[0] - 1))
                comps = np.real(eigvecs[:, order[1 : 1 + k]])
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
        self._save_free_energy_bar_plot(prefix)

        # FES
        self._save_fes_array(prefix)

        # Collect structured results
        analysis_results: Dict[str, Any] = {}
        assert self.transition_matrix is not None
        assert self.count_matrix is not None
        analysis_results["msm"] = MSMResult(
            transition_matrix=self.transition_matrix,
            count_matrix=self.count_matrix,
            free_energies=self.free_energies,
            stationary_distribution=self.stationary_distribution,
        )
        if self.fes_data is not None:
            analysis_results["fes"] = FESResult(
                free_energy=self.fes_data["free_energy"],
                xedges=self.fes_data["xedges"],
                yedges=self.fes_data["yedges"],
                cv1_name=self.fes_data["cv1_name"],
                cv2_name=self.fes_data["cv2_name"],
                temperature=self.fes_data["temperature"],
            )
        if self.implied_timescales is not None:
            analysis_results["its"] = self.implied_timescales

        results_file = self.output_dir / "analysis_results.pkl"
        json_file = self.output_dir / "analysis_results.json"
        with results_file.open("wb") as f:
            pickle.dump(analysis_results, f)
        with json_file.open("w") as f:
            json.dump(
                {k: v.to_dict(metadata_only=True) for k, v in analysis_results.items()},
                f,
            )

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

    def _save_free_energy_bar_plot(self, prefix: str) -> None:
        """Save bar plot of state free energies with bootstrap error bars."""
        if self.state_table is None:
            return
        try:
            import matplotlib.pyplot as _plt  # type: ignore
        except Exception:
            logger.warning("matplotlib not available, skipping free energy bar plot")
            return
        fe = self.state_table.get("free_energy_kJ_mol")
        if fe is None:
            return
        err = self.state_table.get("free_energy_error", pd.Series(np.zeros(len(fe))))
        fig, ax = _plt.subplots()
        ax.bar(self.state_table["state_id"], fe, yerr=err, capsize=4)
        ax.set_xlabel("State")
        ax.set_ylabel("Free energy (kJ/mol)")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{prefix}_free_energy_bar.png")
        _plt.close(fig)

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

        res = self.implied_timescales
        lag_times = np.asarray(res.lag_times, dtype=float)
        timescales = np.asarray(res.timescales, dtype=float)
        ts_ci = np.asarray(res.timescales_ci, dtype=float)

        dt_ps = self.time_per_frame_ps or 1.0
        lag_ps = lag_times * dt_ps
        ts_ps = timescales * dt_ps
        ts_ci_ps = ts_ci * dt_ps

        unit_label = "ps"
        scale = 1.0
        if (np.max(lag_ps) >= 1000.0) or (np.max(ts_ps) >= 1000.0):
            unit_label = "ns"
            scale = 1e-3

        lag_plot = lag_ps * scale
        ts_plot = ts_ps * scale
        ts_ci_plot = ts_ci_ps * scale

        plt.figure(figsize=(10, 6))
        for i in range(ts_plot.shape[1]):
            mask = (
                np.isfinite(ts_plot[:, i])
                & np.isfinite(ts_ci_plot[:, i, 0])
                & np.isfinite(ts_ci_plot[:, i, 1])
            )
            if np.any(mask):
                plt.plot(
                    lag_plot[mask],
                    ts_plot[mask, i],
                    "o-",
                    label=f"τ{i+1} ({unit_label})",
                )
                plt.fill_between(
                    lag_plot[mask],
                    ts_ci_plot[mask, i, 0],
                    ts_ci_plot[mask, i, 1],
                    alpha=0.2,
                )
            else:
                plt.plot([], [], label=f"τ{i+1} ({unit_label})")
        plt.plot([], [], " ", label="NaNs indicate unstable eigenvalues at this τ")

        if res.recommended_lag_window is not None:
            start, end = res.recommended_lag_window
            plt.axvspan(
                start * dt_ps * scale,
                end * dt_ps * scale,
                color="gray",
                alpha=0.1,
            )

        plt.xlabel(f"Lag Time ({unit_label})")
        plt.ylabel(f"Implied Timescale ({unit_label})")
        plt.title("Implied Timescales Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_file:
            plt.savefig(
                self.output_dir / f"{save_file}.png", dpi=300, bbox_inches="tight"
            )

        plt.show()

    def plot_implied_rates(self, save_file: Optional[str] = None) -> None:
        """Plot implied rates with confidence intervals."""

        if self.implied_timescales is None:
            raise ValueError("Implied timescales must be computed first")

        res = self.implied_timescales
        lag_times = np.asarray(res.lag_times, dtype=float)
        rates = np.asarray(res.rates, dtype=float)
        rate_ci = np.asarray(res.rates_ci, dtype=float)

        dt_ps = self.time_per_frame_ps or 1.0
        lag_ps = lag_times * dt_ps

        lag_unit = "ps"
        lag_scale = 1.0
        if np.max(lag_ps) >= 1000.0:
            lag_unit = "ns"
            lag_scale = 1e-3

        rate_unit = "1/ps"
        rate_scale = 1.0
        if np.max(rates) < 1e-3:
            rate_unit = "1/ns"
            rate_scale = 1e3

        lag_plot = lag_ps * lag_scale
        rate_plot = rates * rate_scale
        rate_ci_plot = rate_ci * rate_scale

        plt.figure(figsize=(10, 6))
        for i in range(rate_plot.shape[1]):
            mask = (
                np.isfinite(rate_plot[:, i])
                & np.isfinite(rate_ci_plot[:, i, 0])
                & np.isfinite(rate_ci_plot[:, i, 1])
            )
            if np.any(mask):
                plt.plot(
                    lag_plot[mask],
                    rate_plot[mask, i],
                    "o-",
                    label=f"k{i+1} ({rate_unit})",
                )
                plt.fill_between(
                    lag_plot[mask],
                    rate_ci_plot[mask, i, 0],
                    rate_ci_plot[mask, i, 1],
                    alpha=0.2,
                )
            else:
                plt.plot([], [], label=f"k{i+1} ({rate_unit})")
        plt.plot([], [], " ", label="NaNs indicate unstable eigenvalues at this τ")

        if res.recommended_lag_window is not None:
            start, end = res.recommended_lag_window
            plt.axvspan(
                start * dt_ps * lag_scale,
                end * dt_ps * lag_scale,
                color="gray",
                alpha=0.1,
            )

        plt.xlabel(f"Lag Time ({lag_unit})")
        plt.ylabel(f"Rate ({rate_unit})")
        plt.title("Implied Rates")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_file:
            plt.savefig(
                self.output_dir / f"{save_file}.png", dpi=300, bbox_inches="tight"
            )

        plt.show()

    def plot_ck_test(
        self,
        save_file: str = "ck_plot.png",
        n_macrostates: int = 3,
        factors: Optional[List[int]] = None,
    ) -> Optional[Path]:
        """Generate and save a CK (Chapman–Kolmogorov) test plot.

        Preference order:
        1) Use deeptime's ck_test and plotting utilities across lag multiples
        2) Fallback to a macrostate-level bar plot of MSE vs factors
        """
        # Normalize filename
        try:
            out_path = self.output_dir / (
                save_file
                if str(save_file).lower().endswith(".png")
                else f"{save_file}.png"
            )
        except Exception:
            return None

        # Attempt deeptime-based CK plot
        try:
            from deeptime.markov import TransitionCountEstimator  # type: ignore
            from deeptime.markov.msm import MaximumLikelihoodMSM  # type: ignore
            from deeptime.plots import plot_ck_test  # type: ignore
            from deeptime.util.validation import ck_test  # type: ignore

            base_lag = int(max(1, self.lag_time))
            facs = self._normalize_ck_factors(factors)
            lags = [base_lag] + [base_lag * f for f in facs]

            models = []
            for L in lags:
                tce = TransitionCountEstimator(
                    lagtime=int(L), count_mode="sliding", sparse=False
                )
                C = tce.fit(self.dtrajs).fetch_model()
                ml = MaximumLikelihoodMSM(reversible=True)
                models.append(ml.fit(C).fetch_model())

            ckobj = ck_test(models=models, n_metastable_sets=int(max(2, n_macrostates)))
            import matplotlib.pyplot as _plt  # type: ignore

            fig = plot_ck_test(ckobj)
            fig.savefig(out_path, dpi=200)
            _plt.close(fig)
            logger.info(f"Saved CK test plot: {out_path}")
            return out_path
        except Exception:
            pass

        # Fallback: internal CK bar plot
        try:
            facs = self._normalize_ck_factors(factors)
            result = self.compute_ck_test_macrostates(
                n_macrostates=int(max(2, n_macrostates)), factors=facs
            )

            plt.figure(figsize=(7, 5))
            if result.mse:
                xs = sorted(result.mse.keys())
                ys = [result.mse[k] for k in xs]
                plt.bar(xs, ys, color="tab:orange", alpha=0.8, width=0.6)
                plt.xticks(xs, [str(x) for x in xs])
                plt.xlabel("Lag multiple (k)")
                plt.ylabel("MSE(T^k, T_empirical@k·lag)")
                plt.title(f"Chapman–Kolmogorov test ({result.mode}state)")
            else:
                thresh = ", ".join(f"{k}={v}" for k, v in result.thresholds.items())
                plt.text(
                    0.5,
                    0.5,
                    f"insufficient data\n{thresh}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                plt.title("Chapman–Kolmogorov test")
                plt.axis("off")

            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            logger.info(f"Saved CK test plot: {out_path}")
            return out_path
        except Exception:
            return None

    # ---------------- Bayesian MSM uncertainty (deeptime) ----------------

    def sample_bayesian_timescales(
        self, n_samples: int = 200, count_mode: str = "effective"
    ) -> Optional[Dict[str, Any]]:
        """Sample Bayesian MSM posteriors to estimate CIs for ITS and populations.

        Uses effective counts for uncertainty estimation when possible.
        Returns dict with arrays for timescales_samples and population_samples.
        """
        try:
            count_model = self._bmsm_build_counts(count_mode)
            samples_model = self._bmsm_fit_samples(count_model, n_samples)
            ts_list = self._bmsm_collect_timescales(samples_model)
            pi_list = self._bmsm_collect_populations(samples_model)
            if not ts_list and not pi_list:
                return None
            return self._bmsm_finalize_output(ts_list, pi_list)
        except Exception:
            return None

    # ---------------- Helper methods for Bayesian MSM (C901 split) ----------------

    def _bmsm_build_counts(self, count_mode: str) -> Any:
        from deeptime.markov import TransitionCountEstimator  # type: ignore

        tce = TransitionCountEstimator(
            lagtime=int(max(1, self.lag_time)),
            count_mode=str(count_mode),
            sparse=False,
        )
        return tce.fit(self.dtrajs).fetch_model()

    def _bmsm_fit_samples(self, count_model: Any, n_samples: int) -> Any:
        from deeptime.markov.msm import BayesianMSM  # type: ignore

        bmsm = BayesianMSM(reversible=True, n_samples=int(max(1, n_samples)))
        return bmsm.fit(count_model).fetch_model()

    def _bmsm_collect_timescales(self, samples_model: Any) -> List[np.ndarray]:
        ts_list: List[np.ndarray] = []
        for sm in getattr(samples_model, "samples", []):  # type: ignore[attr-defined]
            T = np.asarray(getattr(sm, "transition_matrix", None), dtype=float)
            if T.size == 0:
                continue
            evals = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
            ts = safe_timescales(self.lag_time, evals[1 : min(6, len(evals))])
            ts = ts[np.isfinite(ts)]
            if ts.size:
                ts_list.append(ts)
        return ts_list

    def _bmsm_collect_populations(self, samples_model: Any) -> List[np.ndarray]:
        pi_list: List[np.ndarray] = []
        for sm in getattr(samples_model, "samples", []):  # type: ignore[attr-defined]
            if (
                hasattr(sm, "stationary_distribution")
                and sm.stationary_distribution is not None
            ):
                pi_list.append(np.asarray(sm.stationary_distribution, dtype=float))
        return pi_list

    def _bmsm_finalize_output(
        self, ts_list: List[np.ndarray], pi_list: List[np.ndarray]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if ts_list:
            maxlen = max(arr.shape[0] for arr in ts_list)
            ts_pad = [
                np.pad(a, (0, maxlen - a.shape[0]), constant_values=np.nan)
                for a in ts_list
            ]
            out["timescales_samples"] = np.vstack(ts_pad)
        if pi_list:
            maxn = max(a.shape[0] for a in pi_list)
            pi_pad = [
                (
                    a
                    if a.shape[0] == maxn
                    else np.pad(a, (0, maxn - a.shape[0]), constant_values=np.nan)
                )
                for a in pi_list
            ]
            out["population_samples"] = np.vstack(pi_pad)
        return out

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
    n_states: int | Literal["auto"] = 100,
    lag_time: int = 20,
    feature_type: str = "phi_psi",
    temperatures: Optional[List[float]] = None,
    stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
) -> EnhancedMSM:
    """
    Run complete MSM analysis pipeline.

    Args:
        trajectory_files: Trajectory file(s) to analyze
        topology_file: Topology file (PDB)
        output_dir: Output directory
        n_states: Number of states for clustering or ``"auto"``
        lag_time: Lag time for MSM construction
        feature_type: Type of features to use
        temperatures: Temperatures for TRAM analysis
        stride: Frame stride when loading trajectories
        atom_selection: MDTraj atom selection string or indices
        chunk_size: Frames per chunk when streaming trajectories

    Returns:
        EnhancedMSM object with completed analysis
    """
    msm = _initialize_msm_analyzer(
        trajectory_files, topology_file, temperatures, output_dir
    )
    _load_and_prepare(msm, feature_type, n_states, stride, atom_selection, chunk_size)
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


def _load_and_prepare(
    msm: EnhancedMSM,
    feature_type: str,
    n_states: int | Literal["auto"],
    stride: int,
    atom_selection: str | Sequence[int] | None,
    chunk_size: int,
) -> None:
    msm.load_trajectories(
        stride=stride, atom_selection=atom_selection, chunk_size=chunk_size
    )
    # Save a quick φ/ψ sanity scatter before feature building (helps spot issues early)
    try:
        msm.save_phi_psi_scatter_diagnostics()
    except Exception:
        pass
    msm.compute_features(feature_type=feature_type)
    msm.cluster_features(n_states=n_states)


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
        msm.plot_implied_rates(save_file="implied_rates")
        logger.info("\u2713 Implied rates plot saved")
    except Exception as e:
        logger.warning(f"\u26a0 Implied rates plotting failed: {e}")
    try:
        msm.plot_free_energy_profile(save_file="free_energy_profile")
        logger.info("\u2713 Free energy profile plot saved")
    except Exception as e:
        logger.warning(f"\u26a0 Free energy profile plotting failed: {e}")
    # CK diagnostics: persist JSON and plot (best-effort)
    try:
        macro_ck = None
        micro_ck = None
        try:
            macro_ck = msm.compute_ck_test_macrostates(
                n_macrostates=3, factors=[2, 3, 4]
            )
        except Exception:
            macro_ck = None
        try:
            micro_ck = msm.compute_ck_test_micro(factors=[2, 3, 4])
        except Exception:
            micro_ck = None
        try:
            with open(msm.output_dir / "ck_tests.json", "w", encoding="utf-8") as f:
                json.dump({"macro": macro_ck, "micro": micro_ck}, f, indent=2)
        except Exception:
            pass
        try:
            msm.plot_ck_test(save_file="ck_plot", n_macrostates=3, factors=[2, 3, 4])
            logger.info("\u2713 CK test plot saved")
        except Exception as e:
            logger.warning(f"\u26a0 CK test plotting failed: {e}")
    except Exception:
        pass
