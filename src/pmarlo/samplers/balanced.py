"""Samplers for balanced batch selection across temperatures."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from pmarlo import constants as const
from pmarlo.shards.pair_builder import PairBuilder
from pmarlo.shards.schema import Shard

__all__ = ["BalancedTempSampler"]


class BalancedTempSampler:
    """Draw training pairs across temperatures with optional rare-region upweighting."""

    def __init__(
        self,
        shards_by_temperature: Dict[float, List[Shard]],
        pair_builder: PairBuilder,
        *,
        rare_boost: float = 0.2,
        random_seed: Optional[int] = None,
    ) -> None:
        if rare_boost < 0:
            raise ValueError("rare_boost must be non-negative")
        self.shards_by_temperature = shards_by_temperature
        self.pair_builder = pair_builder
        self.rare_boost = float(rare_boost)
        self._rng = np.random.default_rng(random_seed)
        self._cv_embeddings: Dict[str, np.ndarray] = {}
        self._frame_weights: Dict[str, np.ndarray] = {}
        self._occupancy: Dict[float, Dict[Tuple[float, ...], float]] = {}

    def set_cv_embeddings(self, shard_id: str, embeddings: np.ndarray) -> None:
        """Cache CV embeddings for a shard (shape ``[frames, d]``)."""

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D array")
        self._cv_embeddings[shard_id] = np.asarray(embeddings, dtype=np.float32)

    def set_frame_weights(self, shard_id: str, weights: np.ndarray) -> None:
        """Register per-frame weights (length = number of frames)."""

        arr = np.asarray(weights, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError("weights must be a 1-D array")
        if not np.isfinite(arr).all():
            raise ValueError("weights must be finite")
        total = arr.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self._frame_weights[shard_id] = (arr / total).astype(np.float64)

    def clear_state(self) -> None:
        """Drop cached embeddings, weights and occupancy statistics."""

        self._cv_embeddings.clear()
        self._frame_weights.clear()
        self._occupancy.clear()

    def sample_batch(
        self, pairs_per_temperature: int
    ) -> List[Tuple[Shard, np.ndarray]]:
        if pairs_per_temperature <= 0:
            raise ValueError("pairs_per_temperature must be positive")

        batch: List[Tuple[Shard, np.ndarray]] = []
        for temperature, shards in self.shards_by_temperature.items():
            if not shards:
                continue

            for shard_idx in self._rng.permutation(len(shards)):
                shard = shards[int(shard_idx)]
                pairs = self.pair_builder.make_pairs(shard)
                if pairs.size == 0:
                    continue
                k = min(pairs_per_temperature, pairs.shape[0])
                chosen = self._select_pairs(shard, temperature, pairs, k)
                batch.append((shard, chosen))
                break
        return batch

    def _select_pairs(
        self, shard: Shard, temperature: float, pairs: np.ndarray, k: int
    ) -> np.ndarray:
        if k >= pairs.shape[0]:
            chosen = pairs
        else:
            weights = self._pair_weights(shard, temperature, pairs)
            if weights is None:
                idx = self._rng.choice(pairs.shape[0], size=k, replace=False)
            else:
                idx = self._rng.choice(pairs.shape[0], size=k, replace=False, p=weights)
            chosen = pairs[idx]
        self._update_occupancy(shard, temperature, chosen)
        return chosen

    def _pair_weights(
        self, shard: Shard, temperature: float, pairs: np.ndarray
    ) -> Optional[np.ndarray]:
        n_pairs = pairs.shape[0]
        base = self._frame_weights.get(shard.meta.shard_id)
        if base is not None and base.shape[0] != shard.meta.n_frames:
            raise ValueError(
                "frame weights length mismatch for shard "
                f"{shard.meta.shard_id}: expected {shard.meta.n_frames}, "
                f"got {base.shape[0]}"
            )
        base_weights = (
            base[pairs[:, 0]].astype(np.float64)
            if base is not None
            else np.ones(n_pairs, dtype=np.float64)
        )

        embeddings = self._cv_embeddings.get(shard.meta.shard_id)
        rare_component: Optional[np.ndarray] = None
        if (
            self.rare_boost > 0
            and embeddings is not None
            and embeddings.shape[0] == shard.meta.n_frames
        ):
            occupancy = self._occupancy.setdefault(temperature, {})
            rare_component = np.empty(n_pairs, dtype=np.float64)
            for idx, (i, _) in enumerate(pairs):
                key = self._hash_embedding(embeddings[int(i)])
                occ = occupancy.get(key, 0.0)
                rare_component[idx] = (occ + const.NUMERIC_RARE_EVENT_EPSILON) ** (
                    -self.rare_boost
                )

        weights = base_weights
        if rare_component is not None:
            weights = weights * rare_component

        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return None
        return weights / total

    def _update_occupancy(
        self, shard: Shard, temperature: float, chosen_pairs: np.ndarray
    ) -> None:
        embeddings = self._cv_embeddings.get(shard.meta.shard_id)
        if embeddings is None or embeddings.shape[0] != shard.meta.n_frames:
            return
        base = self._frame_weights.get(shard.meta.shard_id)
        if base is not None and base.shape[0] != shard.meta.n_frames:
            raise ValueError(
                "frame weights length mismatch for shard "
                f"{shard.meta.shard_id}: expected {shard.meta.n_frames}, "
                f"got {base.shape[0]}"
            )
        occupancy = self._occupancy.setdefault(temperature, {})
        for i, _ in chosen_pairs:
            key = self._hash_embedding(embeddings[int(i)])
            increment = float(base[int(i)]) if base is not None else 1.0
            occupancy[key] = occupancy.get(key, 0.0) + increment

    @staticmethod
    def _hash_embedding(vec: np.ndarray, decimals: int = 1) -> Tuple[float, ...]:
        rounded = np.round(vec.astype(np.float64), decimals=decimals)
        return tuple(float(x) for x in rounded)
