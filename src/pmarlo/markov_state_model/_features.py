from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import mdtraj as md
import numpy as np


class FeaturesMixin:
    # Attributes provided by the host class
    trajectories: List[md.Trajectory]
    features: Optional[np.ndarray]
    output_dir: Path
    feature_stride: int
    tica_lag: int
    tica_components: Optional[int]
    raw_frames: int
    effective_frames: int

    def compute_features(
        self,
        feature_type: str = "phi_psi",
        n_features: Optional[int] = None,
        feature_stride: int = 1,
        tica_lag: int = 0,
        tica_components: Optional[int] = None,
    ) -> None:
        logger = getattr(self, "logger", None)
        if logger is None:
            import logging as _logging

            logger = _logging.getLogger("pmarlo")
        logger.info(f"Computing {feature_type} features...")

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
        if self.features is not None:
            logger.info(
                f"Features computed: ({strided_frames}, {self.features.shape[1]})"
            )
            logger.info(
                "Raw frames: %d → %d after feature_stride=%d; effective frames after lag %d: %d",
                self.raw_frames,
                strided_frames,
                self.feature_stride,
                self.tica_lag,
                effective_frames,
            )

    def _compute_features_for_traj(
        self, traj: md.Trajectory, feature_type: str, n_features: Optional[int]
    ) -> np.ndarray:
        ft = feature_type.lower()
        if ft.startswith("universal"):
            try:
                from pmarlo import api as _api  # type: ignore

                method = "vamp"
                if ft.endswith("_tica"):
                    method = "tica"
                elif ft.endswith("_pca"):
                    method = "pca"
                # Persist feature cache per trajectory in output_dir
                from pathlib import Path as _Path

                cache_dir = _Path(getattr(self, "output_dir", ".")) / "feature_cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                metric, _meta = _api.compute_universal_metric(
                    traj,
                    feature_specs=None,
                    align=True,
                    atom_selection="name CA",
                    method=method,
                    lag=int(
                        max(
                            1,
                            getattr(self, "tica_lag", 0)
                            or getattr(self, "lag_time", 10),
                        )
                    ),
                    cache_path=str(cache_dir),
                )
                return metric.reshape(-1, 1)
            except Exception:
                return self._compute_phi_psi_features(traj)
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
        try:
            from pathlib import Path as _Path

            from pmarlo import api  # type: ignore

            cache_dir = _Path(getattr(self, "output_dir", ".")) / "feature_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            X, _cols, _periodic = api.compute_features(
                traj, feature_specs=["phi_psi"], cache_path=str(cache_dir)
            )
            if X.size == 0 or X.shape[1] == 0:
                import logging as _logging

                _logging.getLogger("pmarlo").warning(
                    "No dihedral angles found, using Cartesian coordinates"
                )
                return self._fallback_cartesian_features(traj)
            return np.hstack([np.cos(X), np.sin(X)])
        except Exception:
            phi_angles, _ = md.compute_phi(traj)
            psi_angles, _ = md.compute_psi(traj)
            features: List[np.ndarray] = []
            if phi_angles.shape[1] > 0:
                features.extend([np.cos(phi_angles), np.sin(phi_angles)])
            if psi_angles.shape[1] > 0:
                features.extend([np.cos(psi_angles), np.sin(psi_angles)])
            if features:
                return np.hstack(features)
            import logging as _logging

            _logging.getLogger("pmarlo").warning(
                "No dihedral angles found, using Cartesian coordinates"
            )
            return self._fallback_cartesian_features(traj)
            return self._fallback_cartesian_features(traj)

    def _compute_phi_psi_plus_distance_features(
        self, traj: md.Trajectory, n_distance_features: Optional[int]
    ) -> np.ndarray:
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
        if len(ca_indices) < 2:
            raise ValueError("Insufficient Cα atoms for distance features")
        total_pairs = len(ca_indices) * (len(ca_indices) - 1) // 2
        n_pairs = min(n_features or 200, total_pairs)
        pairs: List[List[int]] = []
        for i in range(0, len(ca_indices), 3):
            for j in range(i + 3, len(ca_indices), 3):
                pairs.append([int(ca_indices[i]), int(ca_indices[j])])
                if len(pairs) >= n_pairs:
                    break
            if len(pairs) >= n_pairs:
                break
        return md.compute_distances(traj, pairs)

    def _compute_contact_features(self, traj: md.Trajectory) -> np.ndarray:
        contacts, _pairs = md.compute_contacts(traj, contacts="all", scheme="ca")
        return contacts

    def _combine_all_features(self, feature_blocks: List[np.ndarray]) -> np.ndarray:
        return np.vstack(feature_blocks)

    # TICA hooks expected to be provided by Estimation/Features integration
    def _maybe_apply_tica(self, n_components_hint: Optional[int], lag: int) -> None:
        try:
            from deeptime.decomposition import TICA as _DT_TICA  # type: ignore
        except Exception:
            if lag > 0 and self.features is not None:
                # Best-effort discard when no TICA available
                from typing import cast

                feats = cast(np.ndarray, self.features)
                drop = int(max(0, lag))
                self.features = feats[drop:]
            return

        if self.features is None or n_components_hint is None:
            return
        n_components = int(max(2, min(5, n_components_hint)))
        Xs: List[np.ndarray] = []
        start = 0
        for traj in self.trajectories:
            end = start + traj.n_frames
            Xs.append(self.features[start:end])
            start = end
        try:
            tica = _DT_TICA(lagtime=int(max(1, lag or 1)), dim=n_components)
            tica_model = tica.fit(Xs).fetch_model()
            Ys = [tica_model.transform(x) for x in Xs]
            combined = np.vstack(Ys) if Ys else self.features
            # Discard initial frames equal to lag to align with correlation time
            drop = int(max(0, lag))
            self.features = combined[drop:]
            # Stash marker to indicate TICA succeeded
            self.tica_components_ = n_components  # type: ignore[attr-defined]
        except Exception:
            # Fallback: do not transform, just discard initial lag frames
            from typing import cast

            feats = cast(np.ndarray, self.features)
            drop = int(max(0, lag))
            self.features = feats[drop:]

    def _fallback_cartesian_features(self, traj: md.Trajectory) -> np.ndarray:
        # Flatten Cartesian coordinates per frame as a simple, always-available feature
        xyz = np.asarray(traj.xyz, dtype=float)  # (n_frames, n_atoms, 3)
        return xyz.reshape(xyz.shape[0], -1)
