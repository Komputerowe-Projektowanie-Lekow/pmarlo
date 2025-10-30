"""Frame reweighting helpers for MSM and FES analysis.

Reweighting helpers for downstream MSM/FES analysis with strict failure
semantics: missing required thermodynamic data (energy) or invalid
normalization (non-finite / non-positive sum) raises a :class:`ValueError`
instead of substituting uniform weights.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Sequence

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


@dataclass(slots=True)
class _TRAMInputs:
    dtrajs: list[np.ndarray]
    bias_matrices: list[np.ndarray]
    ttrajs: list[np.ndarray] | None
    n_markov_states: int | None
    n_therm_states: int | None
    lagtime: int
    count_mode: str
    trajectory_indices: Dict[str, int]
    therm_state_indices: Dict[str, int]


@dataclass(slots=True)
class _CachedWeights:
    weights: np.ndarray
    n_frames: int
    beta_sim: float
    energy_signature: tuple[int, str]
    bias_signature: tuple[int, str] | None
    base_signature: tuple[int, str] | None

    def matches(self, thermo: _SplitThermo) -> bool:
        if self.n_frames != thermo.n_frames:
            return False
        if not math.isclose(self.beta_sim, thermo.beta_sim, rel_tol=1e-12, abs_tol=0.0):
            return False
        energy_signature = _array_signature(thermo.energy)
        bias_signature = _array_signature(thermo.bias)
        base_signature = _array_signature(thermo.base_weights)
        if (
            self.energy_signature != energy_signature
            or self.bias_signature != bias_signature
        ):
            return False

        if self.base_signature is None:
            if base_signature is None:
                return True
            if thermo.base_weights is None:
                return True
            return np.array_equal(thermo.base_weights, self.weights)

        return self.base_signature == base_signature

    @classmethod
    def from_split(cls, thermo: _SplitThermo, weights: np.ndarray) -> "_CachedWeights":
        return cls(
            weights=weights,
            n_frames=thermo.n_frames,
            beta_sim=float(thermo.beta_sim),
            energy_signature=_array_signature(thermo.energy),
            bias_signature=_array_signature(thermo.bias),
            base_signature=_array_signature(thermo.base_weights),
        )


def _array_signature(array: np.ndarray | None) -> tuple[int, str] | None:
    if array is None:
        return None
    data = array
    if data.dtype != np.float64 or not data.flags.c_contiguous:
        data = np.ascontiguousarray(data, dtype=np.float64)
    digest = hashlib.sha256(data.view(np.uint8).tobytes()).hexdigest()
    return (int(data.shape[0]), digest)


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
        self._cache: Dict[str, _CachedWeights] = {}

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
        if chosen_mode == AnalysisReweightMode.TRAM:
            weights = self._compute_tram_weights(dataset, splits)
        else:
            weights = {}
            for split_name, thermo in splits.items():
                cached = self._cache.get(thermo.shard_id)
                if cached is not None and cached.matches(thermo):
                    w = cached.weights
                else:
                    w = self._compute_split_weights(thermo)
                    self._cache[thermo.shard_id] = _CachedWeights.from_split(thermo, w)

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

    def _compute_split_weights(self, thermo: _SplitThermo) -> np.ndarray:
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
        return base.astype(np.float64, copy=False)

    # ------------------------------------------------------------------
    # TRAM handling
    # ------------------------------------------------------------------
    def _compute_tram_weights(
        self,
        dataset: AnalysisDataset,
        splits: Dict[str, _SplitThermo],
    ) -> Dict[str, np.ndarray]:
        tram_inputs = self._extract_tram_inputs(dataset, splits)

        try:
            from deeptime.markov.msm import TRAM, TRAMDataset  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "TRAM mode requires the 'deeptime' package to be installed"
            ) from exc

        tram_dataset = TRAMDataset(
            dtrajs=tram_inputs.dtrajs,
            bias_matrices=tram_inputs.bias_matrices,
            ttrajs=tram_inputs.ttrajs,
            n_therm_states=tram_inputs.n_therm_states,
            n_markov_states=tram_inputs.n_markov_states,
            lagtime=int(max(1, tram_inputs.lagtime)),
            count_mode=str(tram_inputs.count_mode),
        )

        tram = TRAM(
            lagtime=int(max(1, tram_inputs.lagtime)),
            count_mode=str(tram_inputs.count_mode),
            init_strategy="MBAR",
        )
        tram_model = tram.fit(tram_dataset).fetch_model()

        unique_states = sorted(set(tram_inputs.therm_state_indices.values()))
        log_weight_cache: Dict[int, Sequence[np.ndarray]] = {}
        for therm_state in unique_states:
            log_weight_cache[therm_state] = tram_model.compute_sample_weights_log(
                tram_inputs.dtrajs,
                tram_inputs.bias_matrices,
                therm_state=int(therm_state),
            )

        weights: Dict[str, np.ndarray] = {}
        for split_name, thermo in splits.items():
            traj_idx = tram_inputs.trajectory_indices[split_name]
            therm_idx = tram_inputs.therm_state_indices[split_name]
            log_weights = np.asarray(
                log_weight_cache[therm_idx][traj_idx], dtype=np.float64
            )

            if log_weights.shape[0] != thermo.n_frames:
                raise ValueError(
                    "TRAM log-weight length mismatch for split "
                    f"'{thermo.shard_id}': {log_weights.shape[0]} != {thermo.n_frames}"
                )

            # Convert log-weights to normalized weights using log-sum-exp for stability
            max_log = float(np.max(log_weights))
            stable = np.exp(log_weights - max_log)

            base = stable
            if thermo.base_weights is not None:
                if thermo.base_weights.shape[0] != thermo.n_frames:
                    raise ValueError(
                        "Split '{thermo.shard_id}' base_weights length mismatch for TRAM: "
                        f"{thermo.base_weights.shape[0]} != {thermo.n_frames}"
                    )
                base = base * thermo.base_weights

            total = float(np.sum(base, dtype=np.float64))
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(
                    f"Split '{thermo.shard_id}' produced non-finite or non-positive weight sum ({total})"
                )

            normalized = (base / total).astype(np.float64, copy=False)
            weights[split_name] = normalized
            self._cache[thermo.shard_id] = _CachedWeights.from_split(thermo, normalized)
            self._store_split_weights(dataset, split_name, thermo.shard_id, normalized)

        return weights

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

    def _extract_tram_inputs(
        self,
        dataset: AnalysisDataset,
        splits: Dict[str, _SplitThermo],
    ) -> _TRAMInputs:
        if not isinstance(dataset, Mapping):
            raise ValueError(
                "TRAM mode requires dataset mappings with 'tram_dataset' metadata"
            )

        tram_payload = dataset.get("tram_dataset")
        if not isinstance(tram_payload, Mapping):
            tram_payload = dataset.get("tram") if isinstance(dataset, Mapping) else None
        if not isinstance(tram_payload, Mapping):
            raise ValueError(
                "TRAM mode requires a 'tram_dataset' mapping containing dtrajs and bias_matrices"
            )

        raw_dtrajs = tram_payload.get("dtrajs")
        if not isinstance(raw_dtrajs, Sequence) or not raw_dtrajs:
            raise ValueError(
                "TRAM dataset must provide a non-empty sequence of 'dtrajs'"
            )

        dtrajs: list[np.ndarray] = []
        for idx, traj in enumerate(raw_dtrajs):
            arr = np.asarray(traj, dtype=np.int32, order="C")
            if arr.ndim != 1:
                arr = np.reshape(arr, (-1,), order="C")
            if arr.size == 0:
                raise ValueError(f"TRAM dtraj at index {idx} is empty")
            dtrajs.append(arr)

        raw_bias = tram_payload.get("bias_matrices")
        if not isinstance(raw_bias, Sequence) or len(raw_bias) != len(dtrajs):
            raise ValueError(
                "TRAM dataset must provide 'bias_matrices' matching the number of dtrajs"
            )

        bias_matrices: list[np.ndarray] = []
        therm_state_count: int | None = None
        for idx, bias in enumerate(raw_bias):
            arr = np.asarray(bias, dtype=np.float64, order="C")
            if arr.ndim != 2:
                raise ValueError(f"TRAM bias matrix {idx} must be two-dimensional")
            if arr.shape[0] != dtrajs[idx].shape[0]:
                raise ValueError(
                    f"TRAM bias matrix {idx} length mismatch: {arr.shape[0]}"
                    f" != {dtrajs[idx].shape[0]}"
                )
            if therm_state_count is None:
                therm_state_count = int(arr.shape[1])
            elif arr.shape[1] != therm_state_count:
                raise ValueError(
                    "TRAM bias matrices must all share the same number of "
                    "thermodynamic states"
                )
            bias_matrices.append(arr)

        raw_ttrajs = tram_payload.get("ttrajs")
        ttrajs: list[np.ndarray] | None = None
        if raw_ttrajs is not None:
            if not isinstance(raw_ttrajs, Sequence) or len(raw_ttrajs) != len(dtrajs):
                raise ValueError("TRAM 'ttrajs' must align with the provided dtrajs")
            ttrajs = []
            for idx, traj in enumerate(raw_ttrajs):
                arr = np.asarray(traj, dtype=np.int32, order="C")
                if arr.ndim != 1:
                    arr = np.reshape(arr, (-1,), order="C")
                if arr.shape[0] != dtrajs[idx].shape[0]:
                    raise ValueError(
                        f"TRAM ttraj {idx} length mismatch: {arr.shape[0]}"
                        f" != {dtrajs[idx].shape[0]}"
                    )
                ttrajs.append(arr)

        n_markov_states = tram_payload.get("n_markov_states")
        if n_markov_states is not None:
            try:
                n_markov_states = int(n_markov_states)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError("TRAM 'n_markov_states' must be an integer") from exc
            if n_markov_states <= 0:
                raise ValueError("TRAM 'n_markov_states' must be positive")

        n_therm_states = tram_payload.get("n_therm_states")
        if n_therm_states is not None:
            try:
                n_therm_states = int(n_therm_states)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError("TRAM 'n_therm_states' must be an integer") from exc
            if n_therm_states <= 0:
                raise ValueError("TRAM 'n_therm_states' must be positive")
            if therm_state_count is not None and n_therm_states != therm_state_count:
                raise ValueError(
                    "TRAM 'n_therm_states' does not match bias matrix column count"
                )
        else:
            n_therm_states = therm_state_count

        if n_therm_states is None or n_therm_states <= 0:
            raise ValueError(
                "TRAM dataset must define at least one thermodynamic state"
            )

        lagtime_raw = tram_payload.get("lagtime", 1)
        try:
            lagtime = int(lagtime_raw)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("TRAM 'lagtime' must be an integer") from exc
        if lagtime <= 0:
            raise ValueError("TRAM 'lagtime' must be positive")

        count_mode_raw = tram_payload.get("count_mode", "sliding")
        count_mode = str(count_mode_raw)

        dataset_splits = dataset.get("splits")
        if not isinstance(dataset_splits, Mapping):
            raise ValueError(
                "TRAM mode requires dataset 'splits' mapping to resolve indices"
            )

        traj_map = tram_payload.get("trajectory_map") or tram_payload.get(
            "trajectory_indices"
        )
        therm_map = tram_payload.get("therm_state_map") or tram_payload.get(
            "therm_state_indices"
        )

        def _lookup_mapping(mapping: Any, keys: Sequence[str | None]) -> Any:
            if not isinstance(mapping, Mapping):
                return None
            for key in keys:
                if key is None:
                    continue
                if key in mapping:
                    return mapping[key]
            return None

        def _lookup_split_field(split_obj: Any, keys: Sequence[str]) -> Any:
            if not isinstance(split_obj, Mapping):
                return None
            for key in keys:
                if key in split_obj:
                    return split_obj[key]
            tram_info = split_obj.get("tram")
            if isinstance(tram_info, Mapping):
                for key in keys:
                    if key in tram_info:
                        return tram_info[key]
            meta = split_obj.get("meta")
            if isinstance(meta, Mapping):
                for key in keys:
                    if key in meta:
                        return meta[key]
            return None

        def _coerce_index(
            raw_value: Any,
            *,
            label: str,
            upper: int | None,
        ) -> int:
            try:
                idx = int(raw_value)
            except Exception as exc:
                raise ValueError(f"TRAM {label} must be an integer") from exc
            if idx < 0:
                raise ValueError(f"TRAM {label} must be non-negative")
            if upper is not None and idx >= upper:
                raise ValueError(
                    f"TRAM {label} {idx} exceeds available range (upper bound {upper})"
                )
            return idx

        trajectory_indices: Dict[str, int] = {}
        therm_state_indices: Dict[str, int] = {}
        for split_name, thermo in splits.items():
            split_obj = dataset_splits.get(split_name)
            lookup_keys = (split_name, thermo.shard_id)
            traj_raw = _lookup_mapping(traj_map, lookup_keys)
            if traj_raw is None:
                traj_raw = _lookup_split_field(
                    split_obj,
                    (
                        "trajectory_index",
                        "tram_trajectory_index",
                        "trajectory",
                        "traj_index",
                    ),
                )
            if traj_raw is None:
                raise ValueError(
                    f"TRAM trajectory index missing for split '{thermo.shard_id}'"
                )
            traj_idx = _coerce_index(
                traj_raw, label="trajectory index", upper=len(dtrajs)
            )

            therm_raw = _lookup_mapping(therm_map, lookup_keys)
            if therm_raw is None:
                therm_raw = _lookup_split_field(
                    split_obj,
                    (
                        "therm_state_index",
                        "thermodynamic_state_index",
                        "therm_index",
                    ),
                )
            if therm_raw is None:
                raise ValueError(
                    f"TRAM thermodynamic state index missing for split '{thermo.shard_id}'"
                )
            therm_idx = _coerce_index(
                therm_raw, label="thermodynamic state index", upper=n_therm_states
            )

            if splits[split_name].n_frames != dtrajs[traj_idx].shape[0]:
                raise ValueError(
                    f"Split '{thermo.shard_id}' frame count does not match dtraj length: "
                    f"{splits[split_name].n_frames} != {dtrajs[traj_idx].shape[0]}"
                )

            trajectory_indices[split_name] = traj_idx
            therm_state_indices[split_name] = therm_idx

        return _TRAMInputs(
            dtrajs=dtrajs,
            bias_matrices=bias_matrices,
            ttrajs=ttrajs,
            n_markov_states=n_markov_states,
            n_therm_states=n_therm_states,
            lagtime=lagtime,
            count_mode=count_mode,
            trajectory_indices=trajectory_indices,
            therm_state_indices=therm_state_indices,
        )
