from __future__ import annotations

"""TRAM/MBAR reweighting facade for shard-based workflows."""

from typing import Dict, Sequence

import numpy as np

from pmarlo.reweight.reweighter import (
    AnalysisReweightMode,
    Reweighter as AnalysisReweighter,
)
from pmarlo.shards.schema import Shard

from . import (  # noqa: F401  # imported to signal dependency for future integration
    _tram,
)

__all__ = ["Reweighter"]


class Reweighter:
    """Compute per-frame weights for shards relative to a reference temperature."""

    def __init__(self, temperature_ref_K: float) -> None:
        self._analysis = AnalysisReweighter(temperature_ref_K=temperature_ref_K)

    def frame_weights(
        self,
        shards: Sequence[Shard],
        *,
        mode: str = AnalysisReweightMode.MBAR,
    ) -> Dict[str, np.ndarray]:
        """Return per-shard normalized weights."""

        dataset: Dict[str, Dict[str, object]] = {"splits": {}}
        splits = dataset["splits"]
        for shard in shards:
            n_frames = shard.meta.n_frames
            if n_frames <= 0:
                raise ValueError(f"Shard '{shard.meta.shard_id}' has no frames")

            split: Dict[str, object] = {
                "meta": {"shard_id": shard.meta.shard_id},
                "beta": float(shard.meta.beta),
            }

            if shard.energy is not None:
                energy_arr = np.asarray(shard.energy, dtype=np.float64)
                if energy_arr.ndim != 1 or energy_arr.shape[0] != n_frames:
                    raise ValueError(
                        f"Shard '{shard.meta.shard_id}' energy array must be 1-D length {n_frames}"
                    )
                split["energy"] = energy_arr
            else:
                split["energy"] = None

            if shard.bias is not None:
                bias_arr = np.asarray(shard.bias, dtype=np.float64)
                if bias_arr.ndim != 1 or bias_arr.shape[0] != n_frames:
                    raise ValueError(
                        f"Shard '{shard.meta.shard_id}' bias shape mismatch with energy"
                    )
                split["bias"] = bias_arr

            if shard.w_frame is not None:
                base_arr = np.asarray(shard.w_frame, dtype=np.float64)
                if base_arr.ndim != 1 or base_arr.shape[0] != n_frames:
                    raise ValueError(
                        f"Shard '{shard.meta.shard_id}' w_frame length mismatch: {base_arr.shape[0]} != {n_frames}"
                    )
                split["w_frame"] = base_arr

            splits[shard.meta.shard_id] = split

        weights = self._analysis.apply(dataset, mode=mode)
        return {key: np.asarray(val, dtype=np.float64) for key, val in weights.items()}
