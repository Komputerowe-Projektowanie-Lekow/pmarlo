"""Frame reweighting helpers for MSM and FES analysis."""

from __future__ import annotations

"""Reweighting helpers for downstream MSM/FES analysis."""

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping

import math
import numpy as np


_KB_KJ_PER_MOL = 0.00831446261815324


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
    """Compute per-frame analysis weights relative to a reference temperature."""

    def __init__(self, temperature_ref_K: float) -> None:
        if not math.isfinite(temperature_ref_K) or temperature_ref_K <= 0:
            raise ValueError("temperature_ref_K must be a positive finite value")
        self.temperature_ref_K = float(temperature_ref_K)
        self.beta_ref = 1.0 / (_KB_KJ_PER_MOL * self.temperature_ref_K)
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

        When thermodynamic information is not available the returned weights are
        uniform for the respective split.
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
            if hasattr(frame_weights, 'update'):
                frame_weights.update(weights)  # type: ignore[attr-defined]
        return weights

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_splits(
        self, dataset: AnalysisDataset
    ) -> Dict[str, _SplitThermo]:
        splits_raw = dataset.get("splits") if isinstance(dataset, Mapping) else None
        if not isinstance(splits_raw, Mapping):
            return {}

        splits: Dict[str, _SplitThermo] = {}
        for name, split in splits_raw.items():
            shard_id = self._coerce_shard_id(name, split)
            thermo = _SplitThermo(
                shard_id=shard_id,
                beta_sim=self._coerce_beta(split),
                energy=self._coerce_optional_array(split, "energy"),
                bias=self._coerce_optional_array(split, "bias"),
                base_weights=self._coerce_optional_array(split, "weights"),
            )
            splits[str(name)] = thermo
        return splits

    def _coerce_optional_array(
        self, split: object, key: str
    ) -> np.ndarray | None:
        if isinstance(split, Mapping):
            val = split.get(key)
        else:
            val = getattr(split, key, None)
        if val is None:
            return None
        arr = np.asarray(val, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return None
        return arr

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
        beta = None
        temp = None
        if isinstance(split, Mapping):
            beta = split.get("beta")
            temp = split.get("temperature_K")
        else:
            beta = getattr(split, "beta", None)
            temp = getattr(split, "temperature_K", None)
        if beta is not None:
            beta_val = float(beta)
            if beta_val > 0 and math.isfinite(beta_val):
                return beta_val
        if temp is not None:
            T = float(temp)
            if T > 0 and math.isfinite(T):
                return 1.0 / (_KB_KJ_PER_MOL * T)
        raise ValueError(
            "Each split must define beta or temperature_K for reweighting"
        )

    def _compute_split_weights(
        self, thermo: _SplitThermo, mode: str
    ) -> np.ndarray:
        n_frames = thermo.n_frames
        if n_frames <= 0:
            return np.zeros((0,), dtype=np.float64)

        if thermo.energy is None:
            base = np.ones(n_frames, dtype=np.float64)
        else:
            exponent = -(self.beta_ref - thermo.beta_sim) * thermo.energy
            if thermo.bias is not None:
                exponent -= self.beta_ref * thermo.bias
            exponent = np.clip(exponent - np.max(exponent), -700.0, 700.0)
            base = np.exp(exponent, dtype=np.float64)

        if thermo.base_weights is not None:
            base = base * thermo.base_weights

        total = float(np.sum(base))
        if not math.isfinite(total) or total <= 0:
            base = np.ones(n_frames, dtype=np.float64)
            total = float(n_frames)
        weights = base / total

        # ``TRAM`` currently shares the MBAR estimator until multi-ensemble data
        # becomes available.  The explicit branch is kept to make the behaviour
        # obvious during debugging and simplifies future upgrades.
        if mode == AnalysisReweightMode.TRAM and thermo.energy is not None:
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
                split["weights"] = weights
        cache = dataset.setdefault("__weights__", {})
        if isinstance(cache, MutableMapping):
            cache[shard_id] = weights
