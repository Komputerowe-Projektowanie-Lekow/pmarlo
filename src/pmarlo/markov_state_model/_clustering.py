from __future__ import annotations

from typing import Any, List, Literal, Optional

import numpy as np

from .clustering import ClusteringResult, cluster_microstates


class ClusteringMixin:
    # Attributes provided by the host class
    features: Optional[np.ndarray]
    random_state: Optional[int]
    trajectories: List[Any]
    dtrajs: List[np.ndarray]
    cluster_centers: Optional[np.ndarray]

    def cluster_features(
        self,
        n_states: int | Literal["auto"] = "auto",
        algorithm: str = "kmeans",
        random_state: Optional[int] = None,
    ) -> None:
        if self.features is None:
            raise ValueError("Features must be computed before clustering")

        rng = self.random_state if random_state is None else random_state
        import logging as _logging

        _logging.getLogger("pmarlo").info(
            "Clustering features using %s: requested=%s", algorithm, n_states
        )

        method_choice = (
            algorithm if algorithm in ["kmeans", "minibatchkmeans"] else "kmeans"
        )
        try:
            result: ClusteringResult = cluster_microstates(
                self.features,
                method=method_choice,  # type: ignore[arg-type]
                n_states=n_states,
                random_state=rng,
            )
        except ValueError as exc:
            raise ValueError(
                "Clustering failed to define any microstates; check the requested "
                "state count and feature coverage."
            ) from exc
        labels = result.labels
        self.cluster_centers = result.centers

        if labels.size == 0 or int(result.n_states) <= 0:
            raise ValueError(
                "Clustering produced no valid microstates; cannot proceed with MSM "
                "analysis."
            )

        # Split labels back into trajectories
        self.dtrajs = []
        start_idx = 0
        for traj in self.trajectories:
            end_idx = start_idx + traj.n_frames
            self.dtrajs.append(labels[start_idx:end_idx])
            start_idx = end_idx

        self.n_states = int(result.n_states)
        _logging.getLogger("pmarlo").info(
            "Clustering completed: requested=%s, actual=%d%s",
            n_states,
            self.n_states,
            f" ({result.rationale})" if result.rationale else "",
        )
