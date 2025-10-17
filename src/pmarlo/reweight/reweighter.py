"""Frame reweighting helpers for MSM and FES analysis.

Reweighting helpers for downstream MSM/FES analysis with strict failure
semantics: missing required thermodynamic data (energy) or invalid
normalization (non-finite / non-positive sum) raises a :class:`ValueError`
instead of substituting uniform weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping

import numpy as np

from pmarlo import constants as const

AnalysisDataset = Mapping[str, object] | MutableMapping[str, object]


class AnalysisReweightMode:
    """Enumeration of supported analysis reweighting modes."""

    NONE = "none"
    MBAR = "MBAR"
    TRAM = "TRAM"

    @classmethod
    def normalise(cls, value: str | None) -> str:
        if value is None:
            return cls.NONE
        val = str(value).strip().upper()
        if val == "NONE":
            return cls.NONE
        if val == cls.TRAM:
            return cls.TRAM
        return cls.MBAR


@dataclass(slots=True)
class _SplitThermo:
    shard_id: str
    beta_sim: float
    energy: np.ndarray | None
    bias: np.ndarray | None
    base_weights: np.ndarray | None

    @property
    def n_frames(self) -> int:
        if self.energy is not None:
            return int(self.energy.shape[0])
        if self.bias is not None:
            return int(self.bias.shape[0])
        if self.base_weights is not None:
            return int(self.base_weights.shape[0])
        return 0


class Reweighter:
    """Compute per-frame analysis weights relative to a reference temperature.

    Fail-fast semantics:
      * If a split lacks an energy array, reweighting aborts with ``ValueError``.
      * If normalization produces a non-finite or non-positive sum, a ``ValueError`` is raised.
      * Canonical output key: ``w_frame``.
    """

    def __init__(self, temperature_ref_K: float) -> None:
        if not math.isfinite(temperature_ref_K) or temperature_ref_K <= 0:
            raise ValueError("temperature_ref_K must be a positive finite value")
        self.temperature_ref_K = float(temperature_ref_K)
        self.beta_ref = 1.0 / (
            const.BOLTZMANN_CONSTANT_KJ_PER_MOL * self.temperature_ref_K
        )
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(
        self,
        dataset: AnalysisDataset,
        *,
        mode: str = AnalysisReweightMode.MBAR,
    ) -> Dict[str, np.ndarray]:
        """Compute weights for each split and attach them to ``dataset``.

        Raises:
            ValueError: if required thermodynamic data are missing or
                        normalization fails.
        """

        splits = self._extract_splits(dataset)
        if not splits:
            raise ValueError("Dataset must expose 'splits' mapping with CV arrays")

        chosen_mode = AnalysisReweightMode.normalise(mode)
        weights: Dict[str, np.ndarray] = {}

        for split_name, thermo in splits.items():
            cached = self._cache.get(thermo.shard_id)
            if cached is not None and cached.shape[0] == thermo.n_frames:
                w = cached
            else:
                w = self._compute_split_weights(thermo, chosen_mode)
                self._cache[thermo.shard_id] = w

            weights[split_name] = w
            self._store_split_weights(dataset, split_name, thermo.shard_id, w)

        # Attach convenience mapping for downstream MSM/FES helpers
        if isinstance(dataset, MutableMapping):
            frame_weights = dataset.setdefault("frame_weights", {})
            if hasattr(frame_weights, "update"):
                frame_weights.update(weights)  # type: ignore[attr-defined]
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_splits(self, dataset: AnalysisDataset) -> Dict[str, _SplitThermo]:
        splits_raw = dataset.get("splits") if isinstance(dataset, Mapping) else None
        if not isinstance(splits_raw, Mapping):
            return {}

        splits: Dict[str, _SplitThermo] = {}
        for name, split in splits_raw.items():
            shard_id = self._coerce_shard_id(name, split)
            base_w = self._coerce_optional_array(split, "w_frame")
            if base_w is None:
                if self._has_key(split, "weights"):
                    raise ValueError(
                        f"Split '{shard_id}' provides base weights under deprecated key "
                        "'weights'; use 'w_frame' instead"
                    )
            thermo = _SplitThermo(
                shard_id=shard_id,
                beta_sim=self._coerce_beta(split),
                energy=self._coerce_optional_array(split, "energy"),
                bias=self._coerce_optional_array(split, "bias"),
                base_weights=base_w,
            )
            splits[str(name)] = thermo
        return splits

    def _coerce_optional_array(self, split: object, key: str) -> np.ndarray | None:
        if isinstance(split, Mapping):
            val = split.get(key)
        else:
            val = getattr(split, key, None)
        if val is None:
            return None
        arr = np.array(val, dtype=np.float64, copy=False, order="C")
        if arr.ndim != 1:
            arr = np.reshape(arr, (-1,), order="C")
        if arr.size == 0:
            return None
        return arr

    def _has_key(self, split: object, key: str) -> bool:
        if isinstance(split, Mapping):
            return key in split
        return hasattr(split, key)

    def _coerce_shard_id(self, name: str, split: object) -> str:
        candidate = None
        if isinstance(split, Mapping):
            meta = split.get("meta")
            if isinstance(meta, Mapping):
                candidate = meta.get("shard_id") or meta.get("id")
            candidate = candidate or split.get("shard_id") or split.get("id")
        else:
            candidate = getattr(split, "shard_id", None)
        if candidate is None:
            return str(name)
        return str(candidate)

    def _coerce_beta(self, split: object) -> float:
        beta: object | None = None
        temp: object | None = None
        if isinstance(split, Mapping):
            beta = split.get("beta")
            temp = split.get("temperature_K")
        else:
            beta = getattr(split, "beta", None)
            temp = getattr(split, "temperature_K", None)
        if beta is not None:
            beta_val: float = float(beta)
            if beta_val > 0 and math.isfinite(beta_val):
                return float(beta_val)
        if temp is not None:
            T: float = float(temp)
            if T > 0 and math.isfinite(T):
                return float(1.0 / (const.BOLTZMANN_CONSTANT_KJ_PER_MOL * T))
        raise ValueError("Each split must define beta or temperature_K for reweighting")

    def _compute_split_weights(self, thermo: _SplitThermo, mode: str) -> np.ndarray:
        # Fail-fast: energy is required (bias alone insufficient for temperature reweighting)
        if thermo.energy is None:
            raise ValueError(
                f"Split '{thermo.shard_id}' missing required 'energy' array for reweighting"
            )

        n_frames = thermo.n_frames
        if n_frames <= 0:
            raise ValueError(
                f"Split '{thermo.shard_id}' is empty (no frames) and cannot be reweighted"
            )

        energy = thermo.energy
        assert energy is not None
        delta_beta = self.beta_ref - thermo.beta_sim
        base = np.empty_like(energy, dtype=np.float64)
        np.multiply(energy, -delta_beta, out=base, casting="unsafe")
        if thermo.bias is not None:
            if thermo.bias.shape[0] != n_frames:
                raise ValueError(
                    f"Split '{thermo.shard_id}' bias length mismatch: "
                    f"{thermo.bias.shape[0]} != {n_frames}"
                )
            np.subtract(base, self.beta_ref * thermo.bias, out=base)

        max_exponent = float(np.max(base))
        np.subtract(base, max_exponent, out=base)
        np.clip(
            base,
            const.NUMERIC_EXP_CLIP_MIN,
            const.NUMERIC_EXP_CLIP_MAX,
            out=base,
        )
        np.exp(base, out=base)

        if thermo.base_weights is not None:
            if thermo.base_weights.shape[0] != n_frames:
                raise ValueError(
                    f"Split '{thermo.shard_id}' base_weights length mismatch: "
                    f"{thermo.base_weights.shape[0]} != {n_frames}"
                )
            np.multiply(base, thermo.base_weights, out=base)

        total = float(np.sum(base, dtype=np.float64))
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(
                f"Split '{thermo.shard_id}' produced non-finite or non-positive weight sum ({total})"
            )
        np.divide(base, total, out=base)
        weights = base

        if mode == AnalysisReweightMode.TRAM:
            # Placeholder: TRAM identical to MBAR single-ensemble for now.
            return weights.astype(np.float64, copy=False)
        return weights.astype(np.float64, copy=False)

    def _store_split_weights(
        self,
        dataset: AnalysisDataset,
        split_name: str,
        shard_id: str,
        weights: np.ndarray,
    ) -> None:
        if not isinstance(dataset, MutableMapping):
            return
        split_map = dataset.get("splits")
        if isinstance(split_map, MutableMapping):
            split = split_map.get(split_name)
            if isinstance(split, MutableMapping):
                # Write canonical key
                split["w_frame"] = weights
        cache = dataset.setdefault("__weights__", {})
        if isinstance(cache, MutableMapping):
            cache[shard_id] = weights
