from __future__ import annotations

"""Joint REMD<->CV orchestrator coordinating shard ingestion and MSM building."""

import logging
from dataclasses import dataclass
from pathlib import Path
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

from .metrics import GuardrailReport, Metrics

__all__ = ["WorkflowConfig", "JointWorkflow"]

logger = logging.getLogger(__name__)


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
        """Initialise or load the CV model prior to joint iterations (TODO)."""

        # TODO: integrate DeepTICA bootstrap (random/TICA @ T_ref)
        self.cv_model = None
        self.trainer = None

    def iteration(self, i: int) -> Metrics:
        """Perform a single CV training iteration and optionally run guided REMD."""

        shard_jsons = select_shards(
            self.cfg.shards_root, temperature_K=self.cfg.temperature_ref_K
        )
        if not shard_jsons:
            logger.info(
                "No shards found for joint workflow iteration at T=%s K under %s; "
                "returning stub metrics.",
                self.cfg.temperature_ref_K,
                self.cfg.shards_root,
            )
            self.last_new_shards = []
            self.last_guardrails = None
            return Metrics(
                vamp2_val=0.0,
                its_val=0.0,
                ck_error=0.0,
                notes="no shards available",
            )

        shards: Sequence[Shard] = load_shards(shard_jsons)
        frame_weights = self._compute_frame_weights(shards)

        # TODO: plug in real DeepTICA training once trainer integration is complete
        metrics = Metrics(vamp2_val=0.0, its_val=0.0, ck_error=0.0, notes=f"iter {i}")

        if self.remd_callback is not None:
            bias_hook = self._build_bias_hook(shards, frame_weights)
            try:
                new_paths = self.remd_callback(bias_hook, i) or []
            except Exception as exc:
                logger.warning("Guided REMD callback failed: %s", exc)
                new_paths = []
            self.last_new_shards = [Path(p) for p in new_paths]
            if self.last_new_shards:
                logger.info(
                    "Registered %d newly generated shards", len(self.last_new_shards)
                )
        return metrics

    def finalize(self) -> MSMResult:
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

        clustering = cluster_microstates(
            concatenated,
            n_states=self.cfg.n_clusters,
            random_state=None,
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

        ck_errors: Dict[int, float] = {}
        for multiplier in (2, 3):
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
            "fes": fes_artifact,
        }
        self.last_guardrails = self.evaluate_guardrails()
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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
                arr = np.ones(shard.meta.n_frames, dtype=np.float64)
            total = arr.sum()
            if not np.isfinite(total) or total <= 0:
                arr = np.ones(shard.meta.n_frames, dtype=np.float64)
                total = float(shard.meta.n_frames)
            weights.append((arr / total).astype(np.float64))
        return weights

    def _build_bias_hook(
        self,
        shards: Sequence[Shard],
        weights_per_shard: Sequence[np.ndarray],
    ) -> BiasHook:
        if not self._has_cv_transform():
            return self._zero_bias_hook()

        gathered = self._gather_cv_data(shards, weights_per_shard)
        if gathered is None:
            return self._zero_bias_hook()

        centers_fes = self._compute_bias_profile(*gathered)
        if centers_fes is None:
            return self._zero_bias_hook()

        centers, fes = centers_fes
        return self._make_bias_hook(centers, fes)

    def _has_cv_transform(self) -> bool:
        return self.cv_model is not None and hasattr(self.cv_model, "transform")

    def _zero_bias_hook(self) -> BiasHook:
        def _hook(cv_values: np.ndarray) -> np.ndarray:
            return np.zeros((np.asarray(cv_values).shape[0]), dtype=np.float64)

        return _hook

    def _gather_cv_data(
        self,
        shards: Sequence[Shard],
        weights_per_shard: Sequence[np.ndarray],
    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        cv_values: List[np.ndarray] = []
        cv_weights: List[np.ndarray] = []
        for shard, weights in zip(shards, weights_per_shard):
            vals = self._transform_shard_cv(shard)
            if vals is None:
                return None
            cv_values.append(vals)
            cv_weights.append(np.asarray(weights, dtype=np.float64))
        if not cv_values:
            return None
        return cv_values, cv_weights

    def _transform_shard_cv(self, shard: Shard) -> Optional[np.ndarray]:
        assert self.cv_model is not None  # guarded by _has_cv_transform
        try:
            vals = np.asarray(self.cv_model.transform(shard.X), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug(
                "CV transform failed for shard %s: %s", shard.meta.shard_id, exc
            )
            return None
        if vals.shape[0] != shard.meta.n_frames:
            logger.debug(
                "CV transform produced mismatched shape for shard %s",
                shard.meta.shard_id,
            )
            return None
        return vals

    def _compute_bias_profile(
        self, cv_values: List[np.ndarray], cv_weights: List[np.ndarray]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        concat_cv = np.concatenate(cv_values, axis=0)
        concat_w = np.concatenate(cv_weights)
        weight_total = float(concat_w.sum())
        if weight_total <= 0 or concat_cv.size == 0:
            return None
        concat_w = concat_w / weight_total

        coord = concat_cv[:, 0]
        lo, hi = float(np.min(coord)), float(np.max(coord))
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            return None

        bins = np.linspace(lo, hi, 128)
        hist, edges = np.histogram(coord, bins=bins, weights=concat_w)
        if hist.sum() <= 0:
            return None

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
                return np.zeros((0,), dtype=np.float64)
            coord_vals = arr if arr.ndim == 1 else arr[:, 0]
            bias = np.interp(coord_vals, centers, fes, left=fes[0], right=fes[-1])
            return bias.astype(np.float64)

        return _hook

    def evaluate_guardrails(self) -> GuardrailReport:
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

        ck_errors = self._extract_ck_errors()
        ck_threshold = 0.15
        ck_ok = True
        if ck_errors:
            ck_ok = all(float(err) <= ck_threshold for err in ck_errors.values())
            if not ck_ok:
                notes["ck"] = ", ".join(
                    f"k={k}: {float(v):.4f}" for k, v in ck_errors.items()
                )
        else:
            notes.setdefault("ck", "insufficient data")

        report = GuardrailReport(
            vamp2_trend_ok=vamp_ok,
            its_plateau_ok=its_ok,
            ck_threshold_ok=ck_ok,
            notes=notes,
        )
        self.last_guardrails = report
        return report

    def _extract_vamp2_series(self) -> List[float]:
        history = []
        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            for entry in getattr(trainer, "history", []):
                val = entry.get("vamp2") if isinstance(entry, dict) else None
                if val is not None:
                    history.append(float(val))
        model_hist = getattr(getattr(self, "cv_model", None), "training_history", None)
        if isinstance(model_hist, dict):
            for entry in model_hist.get("steps", []):
                if isinstance(entry, dict) and entry.get("vamp2") is not None:
                    history.append(float(entry["vamp2"]))
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

    def _extract_ck_errors(self) -> Dict[int, float]:
        if self.last_artifacts is None:
            return {}
        raw = self.last_artifacts.get("ck_errors", {})
        out: Dict[int, float] = {}
        if isinstance(raw, dict):
            for key, val in raw.items():
                try:
                    out[int(key)] = float(val)
                except Exception:
                    continue
        return out

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
            return np.zeros((0,), dtype=np.float64)

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
