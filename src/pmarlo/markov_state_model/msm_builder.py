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

    def __init__(
        self, tau_steps: int, n_clusters: int, *, random_state: int | None = None
    ):
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

        features, weights_coll = self._prepare_features(Y_list, weights_list)

        # Log projection shapes for each shard after dimension reduction/preparation
        import logging as _logging

        logger = _logging.getLogger("pmarlo")
        for i, projected_array in enumerate(features):
            logger.info(f"Projected shard {i}: projection shape={projected_array.shape}")

        concatenated, concatenated_weights = self._concatenate_data(
            features, weights_coll
        )

        clustering = cluster_microstates(
            concatenated,
            n_states=self.n_clusters,
            random_state=self.random_state,
            n_init=50,
        )

        labels = clustering.labels
        n_states = int(clustering.n_states)
        if n_states <= 0:
            raise ValueError("clustering returned zero states")

        T = np.eye(n_states, dtype=float)
        pi = self._estimate_stationary_distribution(
            labels, concatenated_weights, n_states
        )
        its = np.zeros((self.tau_steps,), dtype=float)
        clusters = labels.copy()

        meta: Dict[str, object] = {
            "tau_steps": self.tau_steps,
            "n_clusters": n_states,
            "rationale": clustering.rationale,
        }

        return MSMResult(T=T, pi=pi, its=its, clusters=clusters, meta=meta)

    def _prepare_features(
        self,
        Y_list: Sequence[np.ndarray],
        weights_list: Optional[Sequence[np.ndarray]] = None,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        if not Y_list:
            raise ValueError("Y_list must contain at least one trajectory array")

        import logging as _logging

        logger = _logging.getLogger("pmarlo")

        features: List[np.ndarray] = []
        weights_coll: List[np.ndarray] = []
        for idx, arr in enumerate(Y_list):
            feature_matrix = np.asarray(arr)

            # Log shape immediately after obtaining the feature array for this shard
            logger.debug(f"Featurized shard {idx}: features shape={feature_matrix.shape}")

            self._validate_feature_matrix(feature_matrix)
            if feature_matrix.size == 0:
                continue
            features.append(feature_matrix)
            if weights_list is not None:
                weights_coll.append(
                    self._prepare_weights(weights_list, idx, feature_matrix)
                )

        if not features:
            raise ValueError("No frames provided for MSM building")

        return features, weights_coll

    def _validate_feature_matrix(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2:
            raise ValueError("Each trajectory must be a 2-D array")

    def _prepare_weights(
        self,
        weights_list: Sequence[np.ndarray],
        index: int,
        feature_matrix: np.ndarray,
    ) -> np.ndarray:
        try:
            raw_weights = weights_list[index]
        except IndexError as exc:  # pragma: no cover - defensive branch
            raise ValueError("weights_list must align with Y_list") from exc

        weights = np.asarray(raw_weights, dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != feature_matrix.shape[0]:
            raise ValueError("weights_list entries must match trajectory length")
        return weights

    def _concatenate_data(
        self, features: Sequence[np.ndarray], weights_coll: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        concatenated = np.concatenate(features, axis=0)
        if weights_coll:
            weights = np.concatenate(weights_coll)
        else:
            weights = np.ones(concatenated.shape[0], dtype=np.float64)
        total = weights.sum()
        if total <= 0:
            raise ValueError("Weights must sum to a positive value")
        normalized = weights / total
        return concatenated, normalized

    def _estimate_stationary_distribution(
        self, labels: np.ndarray, weights: np.ndarray, n_states: int
    ) -> np.ndarray:
        pi = np.zeros((n_states,), dtype=float)
        for state in range(n_states):
            mask = labels == state
            if np.any(mask):
                pi[state] = weights[mask].sum()
        total = pi.sum()
        if total > 0:
            pi /= total
        else:
            pi[:] = 1.0 / n_states
        return pi
