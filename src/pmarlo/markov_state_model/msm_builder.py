from __future__ import annotations

"""Thin facade for MSM construction from precomputed embeddings."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import _ck, _clustering, _estimation, _its, _states  # noqa: F401
from .clustering import cluster_microstates

__all__ = ["MSMResult", "MSMBuilder"]


@dataclass
class MSMResult:
    T: np.ndarray
    pi: np.ndarray
    its: np.ndarray
    clusters: np.ndarray
    meta: Dict[str, object]


class MSMBuilder:
    """Placeholder MSM builder; hooks into full stack in subsequent iterations."""

    def __init__(self, tau_steps: int, n_clusters: int, *, random_state: int | None = None):
        if tau_steps <= 0:
            raise ValueError("tau_steps must be positive")
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        self.tau_steps = int(tau_steps)
        self.n_clusters = int(n_clusters)
        self.random_state = random_state

    def fit(
        self,
        Y_list: Sequence[np.ndarray],
        weights_list: Optional[Sequence[np.ndarray]] = None,
    ) -> MSMResult:
        """Cluster embeddings and return a skeletal MSM result."""

        if not Y_list:
            raise ValueError("Y_list must contain at least one trajectory array")

        features: List[np.ndarray] = []
        weights_coll: List[np.ndarray] = []
        for idx, arr in enumerate(Y_list):
            arr = np.asarray(arr)
            if arr.ndim != 2:
                raise ValueError("Each trajectory must be a 2-D array")
            if arr.size == 0:
                continue
            features.append(arr)
            if weights_list is not None:
                w = np.asarray(weights_list[idx], dtype=np.float64)
                if w.ndim != 1 or w.shape[0] != arr.shape[0]:
                    raise ValueError("weights_list entries must match trajectory length")
                weights_coll.append(w)

        if not features:
            raise ValueError("No frames provided for MSM building")

        concatenated = np.concatenate(features, axis=0)
        if weights_coll:
            concatenated_weights = np.concatenate(weights_coll)
            concatenated_weights = concatenated_weights / concatenated_weights.sum()
        else:
            concatenated_weights = np.ones(concatenated.shape[0], dtype=np.float64)
            concatenated_weights /= concatenated_weights.sum()

        clustering = cluster_microstates(
            concatenated,
            n_states=self.n_clusters,
            random_state=self.random_state,
        )

        labels = clustering.labels
        n_states = int(clustering.n_states)
        if n_states <= 0:
            raise ValueError("clustering returned zero states")

        T = np.eye(n_states, dtype=float)
        pi = np.zeros((n_states,), dtype=float)
        for state in range(n_states):
            mask = labels == state
            if np.any(mask):
                pi[state] = concatenated_weights[mask].sum()
        if pi.sum() > 0:
            pi /= pi.sum()
        else:
            pi[:] = 1.0 / n_states

        its = np.zeros((self.tau_steps,), dtype=float)
        clusters = labels.copy()

        meta: Dict[str, object] = {
            "tau_steps": self.tau_steps,
            "n_clusters": n_states,
            "rationale": clustering.rationale,
        }

        return MSMResult(T=T, pi=pi, its=its, clusters=clusters, meta=meta)
