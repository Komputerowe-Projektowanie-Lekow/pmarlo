from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np


class _HasFESAttrs(Protocol):
    features: Optional[np.ndarray]
    stationary_distribution: Optional[np.ndarray]
    lag_time: int
    dtrajs: List[np.ndarray]
    trajectories: List[Any]
    fes_data: Optional[Dict[str, Any]]

    # Helper methods used by FESMixin
    def _validate_fes_prerequisites(self) -> None: ...

    def _extract_collective_variables(
        self, cv1_name: str, cv2_name: str
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def _map_stationary_to_frame_weights(self) -> np.ndarray: ...

    def _choose_bins(self, total_frames: int, user_bins: int) -> int: ...

    def _align_data_lengths(
        self,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def _compute_weighted_histogram(
        self,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
        bins: int,
        ranges: Optional[List[Tuple[float, float]]] = None,
        smooth_sigma: Optional[float] = None,
        periodic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def _histogram_to_free_energy(
        self, H: np.ndarray, temperature: float
    ) -> np.ndarray: ...

    def _store_fes_result(
        self,
        F: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        cv1_name: str,
        cv2_name: str,
        temperature: float,
    ) -> None: ...


class FESMixin:
    # Ensure attribute type compatibility with MSMBase
    fes_data: Optional[Dict[str, Any]]

    def generate_free_energy_surface(
        self: _HasFESAttrs,
        cv1_name: str = "phi",
        cv2_name: str = "psi",
        bins: int = 50,
        temperature: float = 300.0,
    ) -> Dict[str, Any]:
        import logging as _logging

        _logging.getLogger("pmarlo").info(
            f"Generating free energy surface: {cv1_name} vs {cv2_name}"
        )
        self._validate_fes_prerequisites()
        cv1_data, cv2_data = self._extract_collective_variables(cv1_name, cv2_name)
        frame_weights_array = self._map_stationary_to_frame_weights()
        total_frames = int(len(frame_weights_array))
        cv_points = int(len(cv1_data))
        _logging.getLogger("pmarlo").info(
            f"MSM Analysis data: {cv_points} CV points, {total_frames} trajectory frames"
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
            cv1_data,
            cv2_data,
            frame_weights_array,
            bins,
            ranges,
            smooth_sigma=0.6,
            periodic=(cv1_name == "phi" and cv2_name == "psi"),
        )
        F = self._histogram_to_free_energy(H, temperature)
        self._store_fes_result(F, xedges, yedges, cv1_name, cv2_name, temperature)
        self.fes_data = {
            "free_energy": F,
            "xedges": xedges,
            "yedges": yedges,
            "cv1_name": cv1_name,
            "cv2_name": cv2_name,
            "temperature": temperature,
        }
        return self.fes_data

    def _validate_fes_prerequisites(self: _HasFESAttrs) -> None:
        if self.features is None or self.stationary_distribution is None:
            raise ValueError("Features and MSM must be computed first")

    def _map_stationary_to_frame_weights(self: _HasFESAttrs) -> np.ndarray:
        try:
            from deeptime.markov import TransitionCountEstimator  # type: ignore
            from deeptime.markov.msm import MaximumLikelihoodMSM  # type: ignore

            tce = TransitionCountEstimator(
                lagtime=int(max(1, self.lag_time)), count_mode="sliding", sparse=False
            )
            count_model = tce.fit(self.dtrajs).fetch_model()
            ml = MaximumLikelihoodMSM(reversible=True)
            msm = ml.fit(count_model).fetch_model()
            weights_list = msm.compute_trajectory_weights(self.dtrajs)
            return np.concatenate([np.asarray(w, dtype=float) for w in weights_list])
        except Exception:
            frame_weights: list[float] = []
            station = self.stationary_distribution
            if station is None:
                raise ValueError("Stationary distribution not available")
            for dtraj in self.dtrajs:
                for state in dtraj:
                    frame_weights.append(float(station[state]))
            return np.array(frame_weights)

    def _extract_collective_variables(
        self: _HasFESAttrs, cv1_name: str, cv2_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        import mdtraj as md

        cv1_data: list[float] = []
        cv2_data: list[float] = []

        for traj in self.trajectories:
            if cv1_name == "phi" and cv2_name == "psi":
                phi_angles, _ = md.compute_phi(traj)
                psi_angles, _ = md.compute_psi(traj)
                if phi_angles.size > 0 and psi_angles.size > 0:
                    phi_vec = phi_angles[:, 0] if phi_angles.ndim == 2 else phi_angles
                    psi_vec = psi_angles[:, 0] if psi_angles.ndim == 2 else psi_angles
                    phi_deg = np.degrees(np.array(phi_vec).reshape(-1))
                    psi_deg = np.degrees(np.array(psi_vec).reshape(-1))
                    phi_wrapped = ((phi_deg + 180.0) % 360.0) - 180.0
                    psi_wrapped = ((psi_deg + 180.0) % 360.0) - 180.0
                    cv1_data.extend([float(v) for v in phi_wrapped])
                    cv2_data.extend([float(v) for v in psi_wrapped])
                else:
                    raise ValueError("No phi/psi angles found in trajectory")
            else:
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

    def _align_data_lengths(
        self: _HasFESAttrs,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import logging as _logging

        min_length = min(len(cv1_data), len(cv2_data), len(frame_weights_array))
        if len(cv1_data) != len(frame_weights_array):
            _logging.getLogger("pmarlo").warning(
                (
                    f"Length mismatch: CV data ({len(cv1_data)}) vs weights "
                    f"({len(frame_weights_array)}). Truncating to {min_length} points."
                )
            )
            cv1_data = cv1_data[:min_length]
            cv2_data = cv2_data[:min_length]
            frame_weights_array = frame_weights_array[:min_length]
        return cv1_data, cv2_data, frame_weights_array

    def _compute_weighted_histogram(
        self: _HasFESAttrs,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
        bins: int,
        ranges: Optional[List[Tuple[float, float]]] = None,
        smooth_sigma: Optional[float] = None,
        periodic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from scipy.ndimage import gaussian_filter

        try:
            H, xedges, yedges = np.histogram2d(
                cv1_data,
                cv2_data,
                bins=bins,
                weights=frame_weights_array,
                density=True,
                range=ranges,
            )
            if smooth_sigma and smooth_sigma > 0:
                mode = "wrap" if periodic else "reflect"
                H = gaussian_filter(H, sigma=float(smooth_sigma), mode=mode)
            return H, xedges, yedges
        except Exception as e:
            raise ValueError(f"Could not generate histogram for FES: {e}")

    def _histogram_to_free_energy(
        self: _HasFESAttrs, H: np.ndarray, temperature: float
    ) -> np.ndarray:
        from scipy import constants

        kT = constants.k * temperature * constants.Avogadro / 1000.0
        F = np.full_like(H, np.inf)
        mask = H > 1e-12
        if int(np.sum(mask)) == 0:
            raise ValueError(
                "Histogram too sparse for free energy calculation. Try fewer bins or more data"
            )
        F[mask] = -kT * np.log(H[mask])
        finite_mask = np.isfinite(F)
        if int(np.sum(finite_mask)) == 0:
            raise ValueError(
                "All free energy values are infinite - histogram too sparse"
            )
        F_min = float(np.min(F[finite_mask]))
        F[finite_mask] -= F_min
        return F

    def _store_fes_result(
        self: _HasFESAttrs,
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

    def _choose_bins(self: _HasFESAttrs, total_frames: int, user_bins: int) -> int:
        try:
            if total_frames <= 0:
                return max(40, min(60, user_bins))
            reco = int(max(40, min(60, np.sqrt(total_frames) // 6)))
        except Exception:
            reco = 50
        candidate = max(40, min(60, int(user_bins)))
        return candidate if abs(candidate - reco) <= 5 else reco

    def save_phi_psi_scatter_diagnostics(
        self,
        *,
        max_residues: int = 6,
        exclude_special: bool = True,
        sample_per_residue: int = 2000,
        filename: str = "diagnostics_phi_psi_scatter.png",
    ) -> Optional[Path]:
        # Optional diagnostic; implement minimal no-op to keep API compatible
        return None
