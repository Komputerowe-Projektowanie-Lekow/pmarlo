from __future__ import annotations

"""TRAM/MBAR reweighting facade for shard-based workflows.

Fail-fast policy (aligned with pmarlo.reweight):
  * Energy array is required for each shard (no silent uniform fallback).
  * Normalization must yield a finite, strictly positive sum; otherwise raise.
  * Resulting weights sum to 1.0 and are deterministic.
  * Shards are not mutated (weights returned separately).
"""

from typing import Dict, Sequence

import numpy as np

from pmarlo import constants as const
from pmarlo.shards.schema import Shard

from . import (  # noqa: F401  # imported to signal dependency for future integration
    _tram,
)

__all__ = ["Reweighter"]


class Reweighter:
    """Compute per-frame weights for shards relative to a reference temperature.

    Deterministic MBAR-style single-ensemble reweighting with fail-fast semantics.
    """

    def __init__(self, temperature_ref_K: float) -> None:
        if temperature_ref_K <= 0 or not np.isfinite(temperature_ref_K):
            raise ValueError("temperature_ref_K must be a positive finite value")
        self.temperature_ref_K = float(temperature_ref_K)
        self.beta_ref = 1.0 / (
            const.BOLTZMANN_CONSTANT_KJ_PER_MOL * self.temperature_ref_K
        )

    def frame_weights(self, shards: Sequence[Shard]) -> Dict[str, np.ndarray]:
        """Return per-shard normalized weights.

        Raises:
            ValueError: if a shard lacks energy, has zero/negative frame count, or
                        normalization produces a non-finite / non-positive sum.
        """
        weights: Dict[str, np.ndarray] = {}
        for shard in shards:
            n_frames = shard.meta.n_frames
            if n_frames <= 0:
                raise ValueError(f"Shard '{shard.meta.shard_id}' has no frames")

            energy = shard.energy
            if energy is None:
                raise ValueError(
                    f"Shard '{shard.meta.shard_id}' missing required energy array for reweighting"
                )
            energy_arr = np.asarray(energy, dtype=np.float64)
            if energy_arr.ndim != 1 or energy_arr.shape[0] != n_frames:
                raise ValueError(
                    f"Shard '{shard.meta.shard_id}' energy array must be 1-D length {n_frames}"
                )

            bias = shard.bias
            bias_arr = None
            if bias is not None:
                bias_arr = np.asarray(bias, dtype=np.float64)
                if bias_arr.shape != energy_arr.shape:
                    raise ValueError(
                        f"Shard '{shard.meta.shard_id}' bias shape mismatch with energy"
                    )

            arr = self._boltzmann_weights(
                energy=energy_arr,
                beta_sim=shard.meta.beta,
                bias=bias_arr,
            )

            # Multiply existing per-frame weights if provided
            if shard.w_frame is not None:
                base = np.asarray(shard.w_frame, dtype=np.float64)
                if base.shape != arr.shape:
                    raise ValueError(
                        f"Shard '{shard.meta.shard_id}' w_frame length mismatch: {base.shape[0]} != {arr.shape[0]}"
                    )
                arr = arr * base

            total = float(arr.sum())
            if not np.isfinite(total) or total <= 0.0:
                raise ValueError(
                    f"Shard '{shard.meta.shard_id}' produced non-finite or non-positive weight sum ({total})"
                )
            weights[shard.meta.shard_id] = (arr / total).astype(np.float64, copy=False)
        return weights

    def _boltzmann_weights(
        self,
        *,
        energy: np.ndarray,
        beta_sim: float,
        bias: np.ndarray | None,
    ) -> np.ndarray:
        if energy.ndim != 1:
            raise ValueError("energy must be a 1-D array")
        if bias is not None and bias.shape != energy.shape:
            raise ValueError("bias must match energy shape")

        exponent = -(self.beta_ref - beta_sim) * energy
        if bias is not None:
            exponent -= self.beta_ref * bias
        exponent = np.clip(
            exponent - np.max(exponent),
            const.NUMERIC_EXP_CLIP_MIN,
            const.NUMERIC_EXP_CLIP_MAX,
        )
        return np.exp(exponent, dtype=np.float64)
