from __future__ import annotations

"""Joint REMD<->CV orchestrator coordinating shard ingestion and MSM building."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pmarlo import constants as const
from pmarlo.markov_state_model.clustering import cluster_microstates
from pmarlo.markov_state_model.msm_builder import MSMResult
from pmarlo.markov_state_model.reweighter import Reweighter
from pmarlo.replica_exchange.bias_hook import BiasHook
from pmarlo.shards.assemble import load_shards, select_shards
from pmarlo.shards.pair_builder import PairBuilder
from pmarlo.shards.schema import Shard
from pmarlo.validation.ck_rule import CKConfig, CKDecision, decide_ck

from .metrics import GuardrailReport, Metrics

__all__ = ["WorkflowConfig", "JointWorkflow", "CKGuardrailError"]


logger = logging.getLogger(__name__)


class CKGuardrailError(RuntimeError):
    """Raised when the Chapman–Kolmogorov guardrail fails."""


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for the joint learning workflow orchestrator."""

    shards_root: Path
    temperature_ref_K: float
    tau_steps: int
    n_clusters: int
    use_reweight: bool = True
    artifact_dir: Optional[Path] = None


class JointWorkflow:
    """Coordinate CV training iterations and downstream MSM construction."""

    def __init__(self, cfg: WorkflowConfig) -> None:
        self.cfg = cfg
        self.pair_builder = PairBuilder(cfg.tau_steps)
        self.reweighter: Optional[Reweighter] = (
            Reweighter(cfg.temperature_ref_K) if cfg.use_reweight else None
        )
        self.cv_model = None
        self.trainer = None
        self.last_weights: Dict[str, np.ndarray] | None = None
        self.last_result: Optional[MSMResult] = None
        self.last_artifacts: Dict[str, Any] | None = None
        self.last_new_shards: List[Path] = []
        self.last_guardrails: Optional[GuardrailReport] = None
        self.last_ck_decision: Optional[CKDecision] = None
        self.last_ck_config: Optional[CKConfig] = None
        self.vamp2_history: List[float] = []
        self.remd_callback: Optional[
            Callable[[BiasHook, int], Optional[Sequence[Path]]]
        ] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_remd_callback(
        self, callback: Callable[[BiasHook, int], Optional[Sequence[Path]]]
    ) -> None:
        """Register a callback used to launch guided REMD between iterations.

        The callback receives ``(bias_hook, iteration_index)`` and should return
        an iterable of newly generated shard JSON paths (if any).
        """

        self.remd_callback = callback

    def bootstrap_cv(self) -> None:
        """Initialise the CV model using shards at the reference temperature."""

        shard_jsons = select_shards(
            self.cfg.shards_root, temperature_K=self.cfg.temperature_ref_K
        )
        if not shard_jsons:
            raise ValueError(
                f"No shards found for CV bootstrap at {self.cfg.temperature_ref_K} K "
                f"under {self.cfg.shards_root}"
            )

        shards: Sequence[Shard] = load_shards(shard_jsons)
        frame_weights = self._compute_frame_weights(shards)
        self._bootstrap_from_shards(shards, frame_weights)

    def iteration(self, i: int) -> Metrics:
        """Perform a single CV training iteration and optionally run guided REMD."""

        shard_jsons = select_shards(
            self.cfg.shards_root, temperature_K=self.cfg.temperature_ref_K
        )
        if not shard_jsons:
            raise ValueError(
                f"No shards found for joint workflow iteration at "
                f"T={self.cfg.temperature_ref_K} K under {self.cfg.shards_root}"
            )

        shards: Sequence[Shard] = load_shards(shard_jsons)
        frame_weights = self._compute_frame_weights(shards)

        if not self._has_cv_transform():
            self._bootstrap_from_shards(shards, frame_weights)

        cv_outputs: List[np.ndarray] = []
        for shard in shards:
            transformed = self._transform_shard_cv(shard)
            if transformed.ndim == 1:
                transformed = transformed.reshape(-1, 1)
            elif transformed.ndim != 2:
                raise ValueError("CV outputs must be 1D or 2D per shard")
            cv_outputs.append(np.asarray(transformed, dtype=np.float64))

        pairs, pair_weights = self._prepare_pairs(shards, frame_weights)
        if pairs.size == 0:
            raise ValueError(
                "No usable lagged pairs found; ensure shards contain sufficient frames"
            )

        concatenated_cv = np.concatenate(cv_outputs, axis=0)
        vamp2_val = self._compute_weighted_vamp2(
            concatenated_cv, pairs, pair_weights
        )
        self._record_vamp2_history(vamp2_val)

        kmeans_kwargs = {"n_init": 50}
        clustering = cluster_microstates(
            concatenated_cv,
            n_states=self.cfg.n_clusters,
            random_state=None,
            **kmeans_kwargs,
        )

        labels = clustering.labels
        n_states = int(clustering.n_states)
        if n_states <= 0:
            raise ValueError("Clustering returned zero states")

        clusters_per_shard: List[np.ndarray] = []
        weights_per_shard: List[np.ndarray] = []
        offset = 0
        for shard, weights in zip(shards, frame_weights):
            length = shard.meta.n_frames
            clusters_per_shard.append(labels[offset : offset + length])
            weights_per_shard.append(np.asarray(weights, dtype=np.float64))
            offset += length

        counts = self._compute_counts(
            shards,
            clusters_per_shard,
            weights_per_shard,
            self.cfg.tau_steps,
            n_states,
        )
        T = self._normalize_counts(counts)

        row_sums = counts.sum(axis=1)
        total_weight = row_sums.sum()
        if total_weight > 0:
            pi = row_sums / total_weight
        else:
            pi = np.full((n_states,), 1.0 / n_states, dtype=np.float64)

        dt_ps = float(np.mean([shard.dt_ps for shard in shards]))
        lag_time_ps = float(self.cfg.tau_steps * dt_ps)
        its_array = self._compute_its(T, lag_time_ps)

        ck_config = self._resolve_ck_config(
            ck_mode=None,
            ck_absolute=None,
            ck_min_pass_fraction=None,
            ck_per_lag_cap=None,
            ck_k_steps=None,
            ck_sigma_mult=None,
        )
        self.last_ck_config = ck_config

        ck_errors: Dict[int, float] = {}
        ck_transition_matrices: Dict[int, np.ndarray] = {}
        ck_row_counts: Dict[int, np.ndarray] = {}
        for multiplier in ck_config.k_steps:
            counts_k = self._compute_counts(
                shards,
                clusters_per_shard,
                weights_per_shard,
                self.cfg.tau_steps * multiplier,
                n_states,
            )
            T_actual = self._normalize_counts(counts_k)
            T_pred = np.linalg.matrix_power(T, multiplier)
            ck_errors[multiplier] = float(np.linalg.norm(T_pred - T_actual, ord="fro"))
            ck_transition_matrices[multiplier] = T_actual
            ck_row_counts[multiplier] = counts_k.sum(axis=1)

        meta: Dict[str, Any] = {
            "n_clusters": clustering.n_states,
            "rationale": clustering.rationale,
            "centers": (
                clustering.centers.tolist() if clustering.centers is not None else None
            ),
            "lag_time_ps": lag_time_ps,
            "ck_errors": ck_errors,
        }

        its_val = float(its_array[0]) if its_array.size else 0.0
        ck_error_metric = max(ck_errors.values()) if ck_errors else 0.0
        metrics = Metrics(
            vamp2_val=float(vamp2_val),
            its_val=its_val,
            ck_error=float(ck_error_metric),
            notes=f"iter {i}",
        )

        self.last_result = MSMResult(
            T=T,
            pi=pi,
            its=its_array,
            clusters=labels,
            meta=meta,
        )
        self.last_artifacts = {
            "transition_matrix": T,
            "counts": counts,
            "stationary_distribution": pi,
            "its": its_array,
            "ck_errors": ck_errors,
            "ck_transition_matrices": ck_transition_matrices,
            "ck_row_counts": ck_row_counts,
        }
        self.last_guardrails = None

        if self.remd_callback is not None:
            bias_hook = self._build_bias_hook(shards, frame_weights)
            new_paths = self.remd_callback(bias_hook, i) or []
            self.last_new_shards = [Path(p) for p in new_paths]
            if self.last_new_shards:
                logger.info(
                    "Registered %d newly generated shards", len(self.last_new_shards)
                )
        return metrics

    def finalize(
        self,
        *,
        ck_mode: str | None = None,
        ck_absolute: float | None = None,
        ck_min_pass_fraction: float | None = None,
        ck_per_lag_cap: float | None = None,
        ck_k_steps: tuple[int, ...] | None = None,
        ck_sigma_mult: float | None = None,
    ) -> MSMResult:
        """Reweight shards, build an MSM, and generate diagnostic artifacts."""

        shard_jsons = select_shards(
            self.cfg.shards_root, temperature_K=self.cfg.temperature_ref_K
        )
        if not shard_jsons:
            raise ValueError(
                f"No shards found at T={self.cfg.temperature_ref_K} K in {self.cfg.shards_root}"
            )

        shards: Sequence[Shard] = load_shards(shard_jsons)
        frame_weights = self._compute_frame_weights(shards)

        features_per_shard: List[np.ndarray] = []
        lengths: List[int] = []
        for shard in shards:
            features = np.asarray(shard.X, dtype=np.float32)
            features_per_shard.append(features)
            lengths.append(features.shape[0])

        concatenated = np.concatenate(features_per_shard, axis=0)
        concatenated_weights = np.concatenate(frame_weights)
        concatenated_weights = concatenated_weights / concatenated_weights.sum()

        kmeans_kwargs = {"n_init": 50}
        clustering = cluster_microstates(
            concatenated,
            n_states=self.cfg.n_clusters,
            random_state=None,
            **kmeans_kwargs,
        )

        labels = clustering.labels
        n_states = int(clustering.n_states)
        if n_states <= 0:
            raise ValueError("Clustering returned zero states")

        clusters_per_shard: List[np.ndarray] = []
        weights_per_shard: List[np.ndarray] = []
        offset = 0
        for length in lengths:
            clusters_per_shard.append(labels[offset : offset + length])
            weights_per_shard.append(concatenated_weights[offset : offset + length])
            offset += length

        counts = self._compute_counts(
            shards,
            clusters_per_shard,
            weights_per_shard,
            self.cfg.tau_steps,
            n_states,
        )
        T = self._normalize_counts(counts)

        row_sums = counts.sum(axis=1)
        total_weight = row_sums.sum()
        if total_weight > 0:
            pi = row_sums / total_weight
        else:
            pi = np.full((n_states,), 1.0 / n_states, dtype=np.float64)

        dt_ps = float(np.mean([shard.dt_ps for shard in shards]))
        lag_time_ps = float(self.cfg.tau_steps * dt_ps)
        its_array = self._compute_its(T, lag_time_ps)

        ck_config = self._resolve_ck_config(
            ck_mode=ck_mode,
            ck_absolute=ck_absolute,
            ck_min_pass_fraction=ck_min_pass_fraction,
            ck_per_lag_cap=ck_per_lag_cap,
            ck_k_steps=ck_k_steps,
            ck_sigma_mult=ck_sigma_mult,
        )
        self.last_ck_config = ck_config

        ck_errors: Dict[int, float] = {}
        ck_transition_matrices: Dict[int, np.ndarray] = {}
        ck_row_counts: Dict[int, np.ndarray] = {}
        for multiplier in ck_config.k_steps:
            counts_k = self._compute_counts(
                shards,
                clusters_per_shard,
                weights_per_shard,
                self.cfg.tau_steps * multiplier,
                n_states,
            )
            T_actual = self._normalize_counts(counts_k)
            T_pred = np.linalg.matrix_power(T, multiplier)
            ck_errors[multiplier] = float(np.linalg.norm(T_pred - T_actual, ord="fro"))
            ck_transition_matrices[multiplier] = T_actual
            ck_row_counts[multiplier] = counts_k.sum(axis=1)

        fes_artifact = self._build_fes(concatenated, concatenated_weights)

        meta: Dict[str, Any] = {
            "n_clusters": clustering.n_states,
            "rationale": clustering.rationale,
            "centers": (
                clustering.centers.tolist() if clustering.centers is not None else None
            ),
            "lag_time_ps": lag_time_ps,
            "ck_errors": ck_errors,
            "fes": fes_artifact,
        }

        result = MSMResult(
            T=T,
            pi=pi,
            its=its_array,
            clusters=labels,
            meta=meta,
        )
        self.last_result = result
        self.last_artifacts = {
            "transition_matrix": T,
            "counts": counts,
            "stationary_distribution": pi,
            "its": its_array,
            "ck_errors": ck_errors,
            "ck_transition_matrices": ck_transition_matrices,
            "ck_row_counts": ck_row_counts,
            "fes": fes_artifact,
        }
        self.last_guardrails = self.evaluate_guardrails(ck_config=ck_config)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _bootstrap_from_shards(
        self, shards: Sequence[Shard], frame_weights: Sequence[np.ndarray]
    ) -> None:
        if not shards:
            raise ValueError("Cannot bootstrap CV without at least one shard")

        cv_arrays: List[np.ndarray] = []
        for shard in shards:
            X = np.asarray(shard.X, dtype=np.float64)
            if X.ndim != 2:
                raise ValueError("Shard feature arrays must be 2D for CV bootstrap")
            cv_arrays.append(X)

        mean, scale = self._compute_feature_stats(cv_arrays, frame_weights)
        standardized = [self._standardize_features(arr, mean, scale) for arr in cv_arrays]

        pairs, pair_weights = self._prepare_pairs(shards, frame_weights)
        if pairs.size == 0:
            raise ValueError(
                "No usable lagged pairs found while bootstrapping the CV model"
            )

        concatenated = np.concatenate(standardized, axis=0)
        vamp2_val = self._compute_weighted_vamp2(concatenated, pairs, pair_weights)

        class _WhitenedIdentityCV:
            def __init__(
                self, mean_vec: np.ndarray, scale_vec: np.ndarray, initial_vamp2: float
            ) -> None:
                self.mean = np.asarray(mean_vec, dtype=np.float64)
                self.scale = np.asarray(scale_vec, dtype=np.float64)
                self.training_history = {"steps": [{"vamp2": float(initial_vamp2)}]}

            def transform(self, X: np.ndarray) -> np.ndarray:
                arr = np.asarray(X, dtype=np.float64)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                if arr.shape[1] != self.mean.shape[0]:
                    raise ValueError(
                        "Input feature dimension does not match bootstrap statistics"
                    )
                return (arr - self.mean) / self.scale

        self.cv_model = _WhitenedIdentityCV(mean, scale, vamp2_val)
        self.trainer = SimpleNamespace(history=[{"vamp2": float(vamp2_val)}])
        self.vamp2_history = [float(vamp2_val)]

    def _compute_feature_stats(
        self, features: Sequence[np.ndarray], frame_weights: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        concatenated = np.concatenate(features, axis=0)
        weights = np.concatenate([np.asarray(w, dtype=np.float64) for w in frame_weights])
        if concatenated.shape[0] != weights.shape[0]:
            raise ValueError("Frame weights length must match concatenated features")
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Frame weights must be finite and sum to a positive value")
        w_norm = weights / total
        mean = np.average(concatenated, axis=0, weights=w_norm)
        centered = concatenated - mean
        var = np.average(centered * centered, axis=0, weights=w_norm)
        scale = np.sqrt(var) + const.NUMERIC_MIN_POSITIVE
        return mean.astype(np.float64), scale.astype(np.float64)

    def _standardize_features(
        self, arr: np.ndarray, mean: np.ndarray, scale: np.ndarray
    ) -> np.ndarray:
        if arr.shape[1] != mean.shape[0]:
            raise ValueError("Feature dimension mismatch during standardization")
        return (np.asarray(arr, dtype=np.float64) - mean) / scale

    def _prepare_pairs(
        self, shards: Sequence[Shard], frame_weights: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        pairs_all: List[np.ndarray] = []
        weights_all: List[np.ndarray] = []
        offset = 0
        for shard, weights in zip(shards, frame_weights):
            pairs = self.pair_builder.make_pairs(shard)
            if pairs.size:
                w = np.asarray(weights, dtype=np.float64)
                pair_w = np.sqrt(w[pairs[:, 0]] * w[pairs[:, 1]])
                pairs_all.append(pairs + offset)
                weights_all.append(pair_w)
            offset += shard.meta.n_frames

        if not pairs_all:
            return (
                np.empty((0, 2), dtype=np.int64),
                np.empty((0,), dtype=np.float64),
            )

        merged_pairs = np.vstack(pairs_all).astype(np.int64, copy=False)
        merged_weights = np.concatenate(weights_all).astype(np.float64, copy=False)
        total_weight = float(merged_weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("Pair weights must be finite and positive")
        merged_weights = merged_weights / total_weight
        return merged_pairs, merged_weights

    def _compute_weighted_vamp2(
        self, concatenated_cv: np.ndarray, pairs: np.ndarray, pair_weights: np.ndarray
    ) -> float:
        if pairs.shape[0] == 0:
            raise ValueError("Cannot compute VAMP-2 without at least one lagged pair")
        if pair_weights.shape[0] != pairs.shape[0]:
            raise ValueError("Pair weights must align with provided pairs")
        if concatenated_cv.ndim == 1:
            concatenated_cv = concatenated_cv.reshape(-1, 1)
        A = concatenated_cv[pairs[:, 0]]
        B = concatenated_cv[pairs[:, 1]]
        if A.shape != B.shape:
            raise ValueError("Paired CV arrays must share the same shape")

        w = np.asarray(pair_weights, dtype=np.float64).reshape(-1)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("Pair weights must sum to a positive finite value")
        w = w / total

        mean_A = np.average(A, axis=0, weights=w)
        mean_B = np.average(B, axis=0, weights=w)
        A_c = A - mean_A
        B_c = B - mean_B
        A_std = np.sqrt(np.average(A_c * A_c, axis=0, weights=w)) + const.NUMERIC_MIN_POSITIVE
        B_std = np.sqrt(np.average(B_c * B_c, axis=0, weights=w)) + const.NUMERIC_MIN_POSITIVE
        A_norm = A_c / A_std
        B_norm = B_c / B_std
        corr = np.average(A_norm * B_norm, axis=0, weights=w)
        return float(np.mean(corr * corr))

    def _record_vamp2_history(self, vamp2_val: float) -> None:
        self.vamp2_history.append(float(vamp2_val))
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            history = list(getattr(trainer, "history", []))
            history.append({"vamp2": float(vamp2_val)})
            trainer.history = history

        model = getattr(self, "cv_model", None)
        if model is not None:
            hist = getattr(model, "training_history", None)
            if not isinstance(hist, dict):
                hist = {}
            steps = list(hist.get("steps", []))
            steps.append({"vamp2": float(vamp2_val)})
            hist["steps"] = steps
            try:
                model.training_history = hist
            except Exception:
                logger.debug(
                    "CV model does not expose mutable training_history; skipping vamp2 log"
                )

    def _compute_frame_weights(self, shards: Sequence[Shard]) -> List[np.ndarray]:
        if self.reweighter is not None:
            self.last_weights = self.reweighter.frame_weights(shards)
        else:
            self.last_weights = {
                shard.meta.shard_id: np.ones(shard.meta.n_frames, dtype=np.float64)
                for shard in shards
            }
        weights: List[np.ndarray] = []
        for shard in shards:
            arr = np.asarray(
                self.last_weights.get(shard.meta.shard_id), dtype=np.float64
            )
            if arr.ndim != 1 or arr.shape[0] != shard.meta.n_frames:
                raise ValueError(
                    "Frame weight array shape mismatch for shard "
                    f"{shard.meta.shard_id}"
                )
            total = arr.sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError(
                    "Frame weights must be finite and sum to a positive value"
                )
            weights.append((arr / total).astype(np.float64))
        return weights

    def _build_bias_hook(
        self,
        shards: Sequence[Shard],
        weights_per_shard: Sequence[np.ndarray],
    ) -> BiasHook:
        if not self._has_cv_transform():
            raise RuntimeError(
                "A CV model implementing 'transform' must be configured "
                "before guided REMD callbacks can be used."
            )

        gathered = self._gather_cv_data(shards, weights_per_shard)
        centers_fes = self._compute_bias_profile(*gathered)
        centers, fes = centers_fes
        return self._make_bias_hook(centers, fes)

    def _has_cv_transform(self) -> bool:
        return self.cv_model is not None and hasattr(self.cv_model, "transform")

    def _gather_cv_data(
        self,
        shards: Sequence[Shard],
        weights_per_shard: Sequence[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        cv_values: List[np.ndarray] = []
        cv_weights: List[np.ndarray] = []
        for shard, weights in zip(shards, weights_per_shard):
            vals = self._transform_shard_cv(shard)
            cv_values.append(vals)
            cv_weights.append(np.asarray(weights, dtype=np.float64))
        if not cv_values:
            raise ValueError("No CV data available for bias construction")
        return cv_values, cv_weights

    def _transform_shard_cv(self, shard: Shard) -> np.ndarray:
        assert self.cv_model is not None  # guarded by _has_cv_transform
        vals = np.asarray(self.cv_model.transform(shard.X), dtype=np.float64)
        if vals.shape[0] != shard.meta.n_frames:
            raise ValueError(
                "CV transform produced mismatched frame count for shard "
                f"{shard.meta.shard_id}"
            )
        return vals

    def _compute_bias_profile(
        self, cv_values: List[np.ndarray], cv_weights: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        concat_cv = np.concatenate(cv_values, axis=0)
        concat_cv = np.asarray(concat_cv, dtype=np.float64)
        concat_w = np.concatenate(cv_weights)
        weight_total = float(concat_w.sum())
        if weight_total <= 0 or concat_cv.size == 0:
            raise ValueError("Cannot build bias profile without positive weight")
        concat_w = concat_w / weight_total

        # BUGFIX: handle scalar CV outputs where the transform returns a 1-D array
        if concat_cv.ndim == 1:
            coord = concat_cv
        else:
            coord = concat_cv[:, 0]
        lo, hi = float(np.min(coord)), float(np.max(coord))
        bounds = np.array([lo, hi], dtype=np.float64)
        if not np.all(np.isfinite(bounds)) or hi <= lo:
            raise ValueError("Invalid CV coordinate range for bias profile")

        bins = np.linspace(lo, hi, 128)
        hist, edges = np.histogram(coord, bins=bins, weights=concat_w)
        if hist.sum() <= 0:
            raise ValueError("Unable to compute CV histogram for bias profile")

        prob = hist / hist.sum()
        fes = -(
            const.BOLTZMANN_CONSTANT_KJ_PER_MOL * self.cfg.temperature_ref_K
        ) * np.log(prob + const.NUMERIC_MIN_POSITIVE)
        finite = np.isfinite(fes)
        if finite.any():
            fes = fes - np.min(fes[finite])
        centers = 0.5 * (edges[1:] + edges[:-1])
        return centers, fes

    def _make_bias_hook(self, centers: np.ndarray, fes: np.ndarray) -> BiasHook:
        def _hook(cv_vals: np.ndarray) -> np.ndarray:
            arr = np.asarray(cv_vals, dtype=np.float64)
            if arr.size == 0:
                return np.empty((0,), dtype=np.float64)
            coord_vals = arr if arr.ndim == 1 else arr[:, 0]
            bias = np.interp(coord_vals, centers, fes, left=fes[0], right=fes[-1])
            return bias.astype(np.float64)

        return _hook

    def evaluate_guardrails(
        self,
        *,
        ck_config: CKConfig | None = None,
        ck_mode: str | None = None,
        ck_absolute: float | None = None,
        ck_min_pass_fraction: float | None = None,
        ck_per_lag_cap: float | None = None,
        ck_k_steps: tuple[int, ...] | None = None,
        ck_sigma_mult: float | None = None,
    ) -> GuardrailReport:
        """Evaluate guardrail conditions (VAMP-2 trend, ITS plateau, CK errors)."""

        notes: Dict[str, str] = {}

        vamp_series = self._extract_vamp2_series()
        vamp_ok = True
        if len(vamp_series) >= 2:
            initial = float(vamp_series[0])
            latest = float(vamp_series[-1])
            tolerance = 0.05 * abs(initial) + const.NUMERIC_ABSOLUTE_TOLERANCE
            vamp_ok = latest + tolerance >= initial
            if not vamp_ok:
                notes["vamp2"] = f"latest={latest:.6f} initial={initial:.6f}"
        else:
            notes.setdefault("vamp2", "insufficient data")

        its_vals = self._extract_its_array()
        its_ok = True
        if its_vals.size >= 2:
            diffs = np.diff(its_vals)
            tolerance = (
                0.1 * np.max(np.abs(its_vals)) + const.NUMERIC_ABSOLUTE_TOLERANCE
            )
            its_ok = bool(np.all(diffs >= -tolerance))
            if not its_ok:
                notes["its"] = f"min_diff={float(diffs.min()):.6f}"
        elif its_vals.size == 0:
            notes.setdefault("its", "insufficient data")

        if ck_config is not None and any(
            param is not None
            for param in (
                ck_mode,
                ck_absolute,
                ck_min_pass_fraction,
                ck_per_lag_cap,
                ck_k_steps,
                ck_sigma_mult,
            )
        ):
            raise ValueError(
                "ck_config cannot be combined with individual CK parameter overrides"
            )

        if ck_config is None:
            ck_config = self._resolve_ck_config(
                ck_mode=ck_mode,
                ck_absolute=ck_absolute,
                ck_min_pass_fraction=ck_min_pass_fraction,
                ck_per_lag_cap=ck_per_lag_cap,
                ck_k_steps=ck_k_steps,
                ck_sigma_mult=ck_sigma_mult,
                base=self.last_ck_config,
            )

        self.last_ck_config = ck_config

        P_taus, P_ktaus, row_counts = self._extract_ck_inputs()
        ck_decision: Optional[CKDecision] = None
        ck_ok = True
        if P_ktaus:
            try:
                ck_decision = decide_ck(P_taus, P_ktaus, row_counts, ck_config)
            except ValueError as exc:  # shape mismatch or invalid counts
                message = (
                    "Invalid CK inputs for guardrail evaluation: "
                    f"{exc}. Provide consistent matrices and row counts."
                )
                raise CKGuardrailError(message) from exc

            self.last_ck_decision = ck_decision
            logger.info(
                "CK guardrail decision: %s | details=%s",
                ck_decision.reason,
                ck_decision.per_lag,
            )
            ck_ok = ck_decision.passed
            if not ck_ok:
                notes["ck"] = ", ".join(
                    f"k={k}: err={data['error']:.4f} thr={data['threshold']:.4f}"
                    for k, data in sorted(ck_decision.per_lag.items())
                )
        else:
            self.last_ck_decision = None
            notes.setdefault("ck", "insufficient data")

        report = GuardrailReport(
            vamp2_trend_ok=vamp_ok,
            its_plateau_ok=its_ok,
            ck_threshold_ok=ck_ok,
            notes=notes,
        )
        self.last_guardrails = report
        if ck_decision is not None and not ck_decision.passed:
            raise CKGuardrailError(ck_decision.reason)
        return report

    def _extract_vamp2_series(self) -> List[float]:
        history: List[float] = []
        seen: set[float] = set()

        def _append(val: float | None) -> None:
            if val is None:
                return
            fval = float(val)
            if fval not in seen:
                history.append(fval)
                seen.add(fval)

        for val in getattr(self, "vamp2_history", []):
            _append(val)
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            for entry in getattr(trainer, "history", []):
                val = entry.get("vamp2") if isinstance(entry, dict) else None
                _append(val)
        model_hist = getattr(getattr(self, "cv_model", None), "training_history", None)
        if isinstance(model_hist, dict):
            for entry in model_hist.get("steps", []):
                if isinstance(entry, dict):
                    _append(entry.get("vamp2"))
        return history

    def _extract_its_array(self) -> np.ndarray:
        if (
            self.last_result is not None
            and getattr(self.last_result, "its", None) is not None
        ):
            arr = np.asarray(self.last_result.its, dtype=np.float64)
            return arr[np.isfinite(arr)]
        if self.last_artifacts is not None:
            its = self.last_artifacts.get("its")
            if its is not None:
                arr = np.asarray(its, dtype=np.float64)
                return arr[np.isfinite(arr)]
        return np.asarray([], dtype=np.float64)

    def _extract_ck_inputs(
        self,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        if self.last_artifacts is None:
            return {}, {}, {}

        base = self.last_artifacts.get("transition_matrix")
        if base is None:
            return {}, {}, {}
        P_tau = np.asarray(base, dtype=np.float64)
        if P_tau.ndim != 2 or P_tau.shape[0] != P_tau.shape[1]:
            return {}, {}, {}

        matrices = self.last_artifacts.get("ck_transition_matrices", {})
        counts_raw = self.last_artifacts.get("ck_row_counts", {})
        P_taus: Dict[int, np.ndarray] = {}
        P_ktaus: Dict[int, np.ndarray] = {}
        row_counts: Dict[int, np.ndarray] = {}

        if isinstance(matrices, dict):
            for key, matrix in matrices.items():
                try:
                    k = int(key)
                except Exception:
                    continue
                counts_candidate = (
                    counts_raw.get(key) if isinstance(counts_raw, dict) else None
                )
                if counts_candidate is None and isinstance(counts_raw, dict):
                    counts_candidate = counts_raw.get(k)
                if counts_candidate is None:
                    continue

                Pk = np.asarray(matrix, dtype=np.float64)
                if Pk.shape != P_tau.shape:
                    continue
                counts = np.asarray(counts_candidate, dtype=np.float64)
                if counts.ndim != 1 or counts.shape[0] != P_tau.shape[0]:
                    continue

                P_taus[k] = P_tau
                P_ktaus[k] = Pk
                row_counts[k] = counts

        return P_taus, P_ktaus, row_counts

    def _resolve_ck_config(
        self,
        *,
        ck_mode: str | None,
        ck_absolute: float | None,
        ck_min_pass_fraction: float | None,
        ck_per_lag_cap: float | None,
        ck_k_steps: tuple[int, ...] | None,
        ck_sigma_mult: float | None,
        base: CKConfig | None = None,
    ) -> CKConfig:
        cfg_base = base or self.last_ck_config or CKConfig()

        def _resolve_float_value(
            explicit: float | None, env_name: str, default_value: float, field: str
        ) -> float:
            if explicit is not None:
                try:
                    return float(explicit)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{field} must be a numeric value; received {explicit!r}."
                    ) from exc
            env_val = os.getenv(env_name)
            if env_val is not None and env_val != "":
                try:
                    return float(env_val)
                except ValueError as exc:
                    raise ValueError(
                        f"Environment variable {env_name} must be numeric;"
                        f" received {env_val!r}."
                    ) from exc
            return float(default_value)

        mode_raw = ck_mode or os.getenv("PMARLO_CK_MODE") or cfg_base.mode
        mode_norm = mode_raw.strip().lower()
        if mode_norm not in {"absolute", "ess_adjusted"}:
            raise ValueError(
                "ck_mode must be 'absolute' or 'ess_adjusted'; "
                f"received {mode_raw!r}."
            )
        mode: Mode = "absolute" if mode_norm == "absolute" else "ess_adjusted"

        absolute = _resolve_float_value(
            ck_absolute, "PMARLO_CK_ABSOLUTE", cfg_base.absolute, "ck_absolute"
        )
        if absolute <= 0.0:
            raise ValueError("ck_absolute must be positive.")

        min_pass = _resolve_float_value(
            ck_min_pass_fraction,
            "PMARLO_CK_MIN_PASS_FRACTION",
            cfg_base.min_pass_fraction,
            "ck_min_pass_fraction",
        )
        if not 0.0 <= min_pass <= 1.0:
            raise ValueError("ck_min_pass_fraction must be between 0 and 1 inclusive.")

        per_lag_cap = _resolve_float_value(
            ck_per_lag_cap,
            "PMARLO_CK_PER_LAG_CAP",
            cfg_base.per_lag_cap,
            "ck_per_lag_cap",
        )
        if per_lag_cap <= 0.0:
            raise ValueError("ck_per_lag_cap must be positive.")

        sigma_mult = _resolve_float_value(
            ck_sigma_mult,
            "PMARLO_CK_SIGMA_MULT",
            cfg_base.sigma_mult,
            "ck_sigma_mult",
        )
        if sigma_mult <= 0.0:
            raise ValueError("ck_sigma_mult must be positive.")

        if ck_k_steps is not None:
            steps_candidate = tuple(int(step) for step in ck_k_steps)
        else:
            env_steps = os.getenv("PMARLO_CK_K_STEPS")
            if env_steps:
                tokens = [tok for tok in env_steps.replace(",", " ").split() if tok]
                if not tokens:
                    raise ValueError(
                        "PMARLO_CK_K_STEPS is set but empty; provide integer lag steps."
                    )
                try:
                    steps_candidate = tuple(int(tok) for tok in tokens)
                except ValueError as exc:
                    raise ValueError(
                        "PMARLO_CK_K_STEPS must contain integers separated by spaces or commas."
                    ) from exc
            else:
                steps_candidate = cfg_base.k_steps

        if not steps_candidate:
            raise ValueError("ck_k_steps must contain at least one lag factor >= 2.")

        cleaned_steps = sorted({int(step) for step in steps_candidate})
        if any(step < 2 for step in cleaned_steps):
            raise ValueError("All ck_k_steps values must be integers >= 2.")
        k_steps_tuple = tuple(cleaned_steps)

        return CKConfig(
            mode=mode,
            absolute=float(absolute),
            min_pass_fraction=float(min_pass),
            per_lag_cap=float(per_lag_cap),
            k_steps=k_steps_tuple,
            sigma_mult=float(sigma_mult),
        )

    def _compute_counts(
        self,
        shards: Sequence[Shard],
        clusters_per_shard: Sequence[np.ndarray],
        weights_per_shard: Sequence[np.ndarray],
        tau_steps: int,
        n_states: int,
    ) -> np.ndarray:
        if tau_steps <= 0 or n_states <= 0:
            return np.zeros((n_states, n_states), dtype=np.float64)

        counts = np.zeros((n_states, n_states), dtype=np.float64)
        builder = PairBuilder(max(1, int(tau_steps)))
        for shard, clusters, weights in zip(
            shards, clusters_per_shard, weights_per_shard
        ):
            if clusters.shape[0] != shard.meta.n_frames:
                raise ValueError("Cluster assignments must match shard length")
            pairs = builder.make_pairs(shard)
            if pairs.size == 0:
                continue
            w = np.asarray(weights, dtype=np.float64)
            for i_idx, j_idx in pairs:
                w_pair = float(np.sqrt(w[int(i_idx)] * w[int(j_idx)]))
                counts[int(clusters[int(i_idx)]), int(clusters[int(j_idx)])] += w_pair
        return counts

    def _normalize_counts(self, counts: np.ndarray) -> np.ndarray:
        if counts.size == 0:
            return counts
        row_sums = counts.sum(axis=1, keepdims=True)
        T = np.zeros_like(counts, dtype=np.float64)
        np.divide(counts, row_sums, out=T, where=row_sums > 0)
        zero_rows = np.where(row_sums.squeeze() <= 0)[0]
        for idx in zero_rows:
            T[idx, idx] = 1.0
        return T

    def _compute_its(
        self, T: np.ndarray, lag_time_ps: float, n_times: int = 5
    ) -> np.ndarray:
        if T.size == 0 or lag_time_ps <= 0:
            return np.empty((0,), dtype=np.float64)

        eigvals = np.linalg.eigvals(T.T)
        eigvals = sorted(eigvals, key=lambda x: -abs(x))
        its: List[float] = []
        for lam in eigvals[1:]:
            lam_abs = min(
                max(abs(lam), const.NUMERIC_MIN_POSITIVE),
                1.0 - const.NUMERIC_MIN_POSITIVE,
            )
            its.append(float(-lag_time_ps / np.log(lam_abs)))
            if len(its) >= n_times:
                break
        return np.asarray(its, dtype=np.float64)

    def _build_fes(
        self,
        concatenated: np.ndarray,
        weights: np.ndarray,
        bins: Tuple[int, int] = (64, 64),
    ) -> Optional[Dict[str, Any]]:
        if concatenated.shape[0] == 0 or concatenated.shape[1] < 2:
            return None
        if weights.sum() <= 0:
            return None

        counts, x_edges, y_edges = np.histogram2d(
            concatenated[:, 0], concatenated[:, 1], bins=bins, weights=weights
        )
        total = counts.sum()
        if total <= 0:
            return None
        prob = counts / total
        kT = const.BOLTZMANN_CONSTANT_KJ_PER_MOL * self.cfg.temperature_ref_K
        F = -kT * np.log(prob + const.NUMERIC_MIN_POSITIVE)
        finite = np.isfinite(F)
        if finite.any():
            F = F - np.min(F[finite])
        else:
            F = np.zeros_like(F)
        return {"F": F, "x_edges": x_edges, "y_edges": y_edges}
