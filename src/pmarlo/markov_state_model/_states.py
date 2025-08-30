from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd


class _HasStateAttrs(Protocol):
    stationary_distribution: Optional[np.ndarray]
    n_states: int
    temperatures: List[float]
    trajectories: List[Any]
    output_dir: Path
    dtrajs: List[np.ndarray]
    features: Optional[np.ndarray]
    transition_matrix: Optional[np.ndarray]
    state_table: Optional[pd.DataFrame]

    # Methods used within StatesMixin
    def _count_frames_per_state(self) -> tuple[np.ndarray, int]: ...

    def _find_representatives(
        self,
    ) -> tuple[List[tuple[int, int]], List[Optional[np.ndarray]]]: ...

    def create_state_table(self) -> pd.DataFrame: ...


class StatesMixin:
    def create_state_table(self: _HasStateAttrs) -> pd.DataFrame:
        import logging as _logging

        _logging.getLogger("pmarlo").info("Creating state summary table...")
        if self.stationary_distribution is None:
            raise ValueError("MSM must be built before creating state table")
        state_data: Dict[str, Any] = {"state_id": range(self.n_states)}
        frame_counts, total_frames = self._count_frames_per_state()
        state_data["counts"] = frame_counts.astype(int)
        population = frame_counts / max(total_frames, 1)
        state_data["population"] = population
        from scipy import constants

        kT = constants.k * float(self.temperatures[0]) * constants.Avogadro / 1000.0
        free_from_pop = -kT * np.log(np.clip(population, 1e-12, None))
        state_data["free_energy_kJ_mol"] = free_from_pop
        representative_frames, _ = self._find_representatives()
        rep_traj_array = np.array([int(rf[0]) for rf in representative_frames])
        rep_frame_array = np.array([int(rf[1]) for rf in representative_frames])
        state_data["representative_traj"] = rep_traj_array
        state_data["representative_frame"] = rep_frame_array
        self.state_table = pd.DataFrame(state_data)
        return self.state_table

    def extract_representative_structures(self: _HasStateAttrs, save_pdb: bool = True):
        import logging as _logging

        _logging.getLogger("pmarlo").info("Extracting representative structures...")
        if self.state_table is None:
            self.create_state_table()
        representative_structures = []
        if self.state_table is not None:
            for _, row in self.state_table.iterrows():
                try:
                    traj_idx = int(row["representative_traj"])
                    frame_idx = int(row["representative_frame"])
                    state_id = int(row["state_id"])
                    if traj_idx >= 0 and frame_idx >= 0:
                        if traj_idx >= len(self.trajectories):
                            _logging.getLogger("pmarlo").warning(
                                f"Invalid trajectory index {traj_idx} for state {state_id}"
                            )
                            continue
                        traj = self.trajectories[traj_idx]
                        if frame_idx >= len(traj):
                            _logging.getLogger("pmarlo").warning(
                                f"Invalid frame index {frame_idx} for state {state_id}"
                            )
                            continue
                        frame = traj[frame_idx]
                        representative_structures.append((state_id, frame))
                        if save_pdb:
                            output_file = (
                                self.output_dir
                                / f"state_{state_id:03d}_representative.pdb"
                            )
                            frame.save_pdb(str(output_file))
                except (ValueError, TypeError, IndexError) as e:
                    _logging.getLogger("pmarlo").warning(
                        f"Error extracting representative structure for state {state_id}: {e}"
                    )
                    continue
        _logging.getLogger("pmarlo").info(
            f"Extracted {len(representative_structures)} representative structures"
        )
        return representative_structures

    def _count_frames_per_state(self: _HasStateAttrs) -> tuple[np.ndarray, int]:
        frame_counts = np.zeros(self.n_states)
        total_frames = 0
        for dtraj in self.dtrajs:
            for state in dtraj:
                frame_counts[state] += 1
                total_frames += 1
        return frame_counts, total_frames

    def _bootstrap_free_energy_errors(
        self: _HasStateAttrs, counts: np.ndarray, n_boot: int = 200
    ) -> np.ndarray:
        if not self.dtrajs:
            return np.zeros(self.n_states)
        assignments = np.concatenate(self.dtrajs)
        rng = np.random.default_rng()
        samples = np.empty((n_boot, self.n_states), dtype=float)
        for i in range(n_boot):
            resample = rng.choice(assignments, size=assignments.size, replace=True)
            samples[i] = np.bincount(resample, minlength=self.n_states)
        from scipy import constants

        kT = constants.k * float(self.temperatures[0]) * constants.Avogadro / 1000.0
        fe_samples = -kT * np.log(np.clip(samples / assignments.size, 1e-12, None))
        return np.nanstd(fe_samples, axis=0)

    def _find_representatives(
        self: _HasStateAttrs,
    ) -> tuple[List[tuple[int, int]], List[Optional[np.ndarray]]]:
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

    def _pcca_lumping(
        self: _HasStateAttrs, n_macrostates: int = 4
    ) -> Optional[np.ndarray]:
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
                from sklearn.cluster import MiniBatchKMeans

                rng = getattr(self, "random_state", 42)
                km = MiniBatchKMeans(n_clusters=n_macrostates, random_state=rng)
                labels = km.fit_predict(comps)
                return labels.astype(int)
            except Exception:
                return None
