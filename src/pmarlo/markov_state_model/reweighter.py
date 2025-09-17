from __future__ import annotations

"""TRAM/MBAR reweighting facade for shard-based workflows."""

from typing import Dict, Sequence

import numpy as np

from pmarlo.shards.schema import Shard

from . import _tram  # noqa: F401  # imported to signal dependency for future integration

__all__ = ["Reweighter"]


_KB_KJ_PER_MOL = 0.00831446261815324


class Reweighter:
    """Compute per-frame weights for shards relative to a reference temperature."""

    def __init__(self, temperature_ref_K: float) -> None:
        if temperature_ref_K <= 0:
            raise ValueError("temperature_ref_K must be positive")
        self.temperature_ref_K = float(temperature_ref_K)
        self.beta_ref = 1.0 / (_KB_KJ_PER_MOL * self.temperature_ref_K)

    def frame_weights(self, shards: Sequence[Shard]) -> Dict[str, np.ndarray]:
        """Return MBAR-style weights for each shard.

        Uniform weights are used when energetic information is unavailable.
        Energies are assumed to be expressed in kJ/mol.
        """

        weights: Dict[str, np.ndarray] = {}
        for shard in shards:
            n_frames = shard.meta.n_frames
            energy = shard.energy
            bias = shard.bias
            frame_weight = shard.w_frame

            if energy is None:
                arr = np.ones(n_frames, dtype=np.float64)
            else:
                arr = self._boltzmann_weights(
                    energy=np.asarray(energy, dtype=np.float64),
                    beta_sim=shard.meta.beta,
                    bias=None if bias is None else np.asarray(bias, dtype=np.float64),
                )

            if frame_weight is not None:
                arr = arr * np.asarray(frame_weight, dtype=np.float64)

            total = arr.sum()
            if not np.isfinite(total) or total <= 0:
                arr = np.ones(n_frames, dtype=np.float64)
                total = float(n_frames)
            weights[shard.meta.shard_id] = (arr / total).astype(np.float64)
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

        exponent = np.clip(exponent - np.max(exponent), -700, 700)
        return np.exp(exponent)
