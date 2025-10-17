from __future__ import annotations

"""Generic experiment runner utilities for the app_usecase workflows.

Each experiment script (E0/E1/E2) imports :func:`run_experiment_cli` with a
fixed experiment name so we only maintain the orchestration logic once.
"""

import argparse
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from pmarlo.utils.path_utils import ensure_directory

from ..backend import BuildConfig, WorkflowBackend
from ..headless import _make_layout
from .common import ExperimentBundle, load_bundle

try:
    from pmarlo.markov_state_model.reweighter import Reweighter as ShardReweighter
except ImportError:  # pragma: no cover - defensive guard
    ShardReweighter = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

DEFAULT_FES_BINS = 72


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _sanitize(obj: Any) -> Any:
    """Recursively convert arbitrary objects into JSON-serialisable structures."""

    if obj is None or isinstance(obj, (bool, int, float, str)):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]
    return str(obj)


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize(payload), handle, indent=2, sort_keys=True)
    return path


def _determine_bins(shards: Sequence[Any], config: Mapping[str, Any]) -> Dict[str, int]:
    if not shards:
        raise ValueError("No shards available to derive bin configuration")
    columns = list(getattr(shards[0].meta.feature_spec, "columns", ()))
    if len(columns) < 2:
        raise ValueError("Shards expose fewer than two CV columns; cannot build FES bins")

    outputs = config.get("outputs", {}) if isinstance(config, Mapping) else {}
    explicit_bins = outputs.get("bins") if isinstance(outputs, Mapping) else None
    default_bins = int(outputs.get("default_bins", DEFAULT_FES_BINS))

    bins: Dict[str, int] = {}
    for idx, name in enumerate(columns[:2]):
        if isinstance(explicit_bins, Mapping) and name in explicit_bins:
            bins[name] = int(explicit_bins[name])
        else:
            bins[name] = default_bins
    return bins


def _extract_seed(transform_cfg: Mapping[str, Any]) -> int:
    seeds = transform_cfg.get("seeds")
    if isinstance(seeds, Mapping):
        for key in ("analysis", "global", "shuffle", "deeptica"):
            if key in seeds:
                try:
                    return int(seeds[key])
                except (TypeError, ValueError):
                    continue
    return 2025


def _resolve_deeptica(transform_cfg: Mapping[str, Any]) -> tuple[bool, Dict[str, Any] | None]:
    deeptica_cfg = transform_cfg.get("deeptica")
    if not isinstance(deeptica_cfg, MutableMapping):
        return False, None
    cfg = dict(deeptica_cfg)
    enabled = bool(cfg.pop("enabled", True))

    if "lag_fallback" in cfg:
        fallback = cfg["lag_fallback"]
        cleaned: list[int] = []
        if isinstance(fallback, (list, tuple)):
            for entry in fallback:
                try:
                    candidate = int(entry)
                except (TypeError, ValueError):
                    continue
                if candidate > 0:
                    cleaned.append(candidate)
        elif fallback is not None:
            try:
                candidate = int(fallback)
            except (TypeError, ValueError):
                candidate = None
            if candidate and candidate > 0:
                cleaned.append(int(candidate))
        if cleaned:
            cfg["lag_fallback"] = cleaned
        else:
            cfg.pop("lag_fallback", None)

    if "skip_on_failure" in cfg:
        cfg["skip_on_failure"] = bool(cfg["skip_on_failure"])

    if "min_pairs" in cfg:
        try:
            cfg["min_pairs"] = int(cfg["min_pairs"])
        except (TypeError, ValueError):
            cfg.pop("min_pairs", None)

    return enabled, cfg if cfg else None


def _normalize_reweight_mode(mode: str | None) -> str:
    if mode is None:
        return "IDENTITY"
    value = str(mode).strip().upper()
    if value in {"MBAR", "TRAM"}:
        return value
    if value == "NONE":
        return "IDENTITY"
    return "IDENTITY" if value == "" else value


def _compute_uniform_weights(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Shard length must be positive for uniform weights")
    return np.full(length, 1.0 / float(length), dtype=np.float64)


@dataclass(slots=True)
class WeightSummary:
    status: str
    info: Dict[str, Any]
    violations: list[Dict[str, Any]]

    @property
    def passed(self) -> bool:
        return self.status == "ok"


def _compute_weights_summary(
    bundle: ExperimentBundle,
    shards: Sequence[Any],
    reweight_cfg: Mapping[str, Any],
) -> WeightSummary:
    mode_raw = reweight_cfg.get("mode", "identity")
    mode = _normalize_reweight_mode(mode_raw)
    ref_temp = float(bundle.manifest.get("reference_temperature_K", 0.0))
    notes = reweight_cfg.get("notes")
    guardrails = reweight_cfg.get("guardrails", {})
    ladders = reweight_cfg.get("ladders", {})
    diagnostics = reweight_cfg.get("diagnostics", {})

    info: Dict[str, Any] = {
        "mode": mode,
        "reference_temperature_K": ref_temp if ref_temp else None,
        "notes": notes,
        "guardrail_thresholds": guardrails,
        "ladder_config": ladders,
        "diagnostics": diagnostics,
    }
    violations: list[Dict[str, Any]] = []

    if not shards:
        raise ValueError("No shards provided for weight summary")
    total_frames = sum(int(shard.meta.n_frames) for shard in shards)
    if total_frames <= 0:
        raise ValueError("Total frame count must be positive")

    weights_by_shard: Dict[str, np.ndarray] = {}
    try:
        if mode in {"IDENTITY", "NONE"}:
            for shard in shards:
                shard_id = str(shard.meta.shard_id)
                weights_by_shard[shard_id] = _compute_uniform_weights(int(shard.meta.n_frames))
        else:
            if ShardReweighter is None:
                raise RuntimeError("pmarlo.markov_state_model.reweighter is unavailable")
            if ref_temp <= 0:
                raise ValueError("reference_temperature_K must be > 0 for reweighting")
            reweighter = ShardReweighter(ref_temp)
            weights_by_shard = reweighter.frame_weights(shards)
    except Exception as exc:
        info["error"] = str(exc)
        violations.append({"code": "reweight_error", "message": str(exc)})
        info["violations"] = violations
        return WeightSummary(status="error", info=info, violations=violations)

    per_shard_summary: list[Dict[str, Any]] = []
    global_arrays: list[np.ndarray] = []
    for shard in shards:
        shard_id = str(shard.meta.shard_id)
        weights = np.asarray(weights_by_shard.get(shard_id), dtype=np.float64)
        if weights.size != int(shard.meta.n_frames):
            raise ValueError(f"Weight array length mismatch for shard {shard_id}")
        mean_val = float(np.mean(weights))
        std_val = float(np.std(weights))
        coef_var = float(std_val / mean_val) if mean_val else float("nan")
        ess_val = float(1.0 / np.sum(np.square(weights)))
        ess_fraction = ess_val / float(weights.size) if weights.size else 0.0
        scaled = weights * (float(weights.size) / float(total_frames))
        global_arrays.append(scaled)
        per_shard_summary.append(
            {
                "shard_id": shard_id,
                "frames": int(weights.size),
                "mean": mean_val,
                "std": std_val,
                "coefficient_of_variation": coef_var,
                "min": float(np.min(weights)),
                "max": float(np.max(weights)),
                "ess": ess_val,
                "ess_fraction": ess_fraction,
                "global_share": float(np.sum(scaled)),
            }
        )
        if mode in {"IDENTITY", "NONE"} and coef_var > 0.05:
            violations.append(
                {
                    "code": "non_uniform_identity_weights",
                    "shard_id": shard_id,
                    "coefficient_of_variation": coef_var,
                }
            )

    concat = np.concatenate(global_arrays)
    global_ess = float(1.0 / np.sum(np.square(concat)))
    global_ess_fraction = float(global_ess / float(total_frames)) if total_frames else 0.0
    info.update(
        {
            "total_frames": total_frames,
            "global_ess": global_ess,
            "global_ess_fraction": global_ess_fraction,
            "per_shard": per_shard_summary,
        }
    )

    ess_threshold = guardrails.get("min_effective_sample_fraction")
    if ess_threshold is not None:
        try:
            ess_threshold_val = float(ess_threshold)
        except (TypeError, ValueError):
            ess_threshold_val = None
        if ess_threshold_val is not None and global_ess_fraction < ess_threshold_val:
            violations.append(
                {
                    "code": "ess_fraction_below_threshold",
                    "threshold": ess_threshold_val,
                    "actual": global_ess_fraction,
                }
            )

    status = "ok" if not violations else "fail"
    info["violations"] = violations
    return WeightSummary(status=status, info=info, violations=violations)


def _build_config(bundle: ExperimentBundle, shards: Sequence[Any]) -> BuildConfig:
    transform_cfg = bundle.configs.transform
    discretize_cfg = bundle.configs.discretize
    reweight_cfg = bundle.configs.reweighter

    bins = _determine_bins(shards, transform_cfg)
    lag = int(discretize_cfg.get("lag_steps", 6))
    seed = _extract_seed(transform_cfg)
    temperature = float(bundle.manifest.get("reference_temperature_K", 300.0))
    learn_cv, deeptica_params = _resolve_deeptica(transform_cfg)
    preprocess = transform_cfg.get("preprocessing", {}) if isinstance(transform_cfg, Mapping) else {}
    cluster_mode = discretize_cfg.get("engine") or discretize_cfg.get("mode") or "kmeans"
    n_microstates = int(discretize_cfg.get("n_clusters", 150))
    fes_cfg = transform_cfg.get("outputs", {}) if isinstance(transform_cfg, Mapping) else {}

    notes = {
        "experiment": bundle.layout.name,
        "manifest_path": str(bundle.manifest_path),
        "reweight_mode_requested": _normalize_reweight_mode(reweight_cfg.get("mode")),
        "config_snapshot": {
            "transform": transform_cfg,
            "discretize": discretize_cfg,
            "reweighter": reweight_cfg,
            "msm": bundle.configs.msm,
        },
    }

    return BuildConfig(
        lag=lag,
        bins=bins,
        seed=seed,
        temperature=temperature,
        learn_cv=learn_cv,
        deeptica_params=deeptica_params,
        notes=_sanitize(notes),
        apply_cv_whitening=bool(preprocess.get("whitening", True)),
        cluster_mode=str(cluster_mode),
        n_microstates=n_microstates,
        reweight_mode=_normalize_reweight_mode(reweight_cfg.get("mode")),
        fes_method=str(fes_cfg.get("method", "kde")),
        fes_bandwidth=fes_cfg.get("bandwidth", "scott"),
        fes_min_count_per_bin=int(fes_cfg.get("min_count", 1)),
    )


def _collect_debug_outputs(artifact: Any, output_dir: Path) -> Dict[str, Any]:
    analysis_debug_dir = output_dir / "analysis_debug"
    ensure_directory(analysis_debug_dir)

    summary_payload = dict(artifact.debug_summary or {})
    summary_path = analysis_debug_dir / "summary.json"
    _write_json(summary_path, summary_payload)

    # If export directory differs, replicate summary entry for easy discovery
    if artifact.debug_dir:
        src_summary = Path(artifact.debug_dir) / "summary.json"
        if src_summary.exists():
            target_dir = analysis_debug_dir / Path(artifact.debug_dir).name
            ensure_directory(target_dir)
            dest = target_dir / "summary.json"
            try:
                shutil.copy2(src_summary, dest)
            except Exception:
                LOGGER.debug("Failed to copy debug summary from %s to %s", src_summary, dest, exc_info=True)

    return {
        "summary_path": str(summary_path),
        "analysis_debug_dir": str(analysis_debug_dir),
        "tau_frames": artifact.tau_frames,
        "effective_tau_frames": artifact.effective_tau_frames,
        "effective_stride_max": artifact.effective_stride_max,
    }


def _relativize(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _collect_artifact_paths(
    output_dir: Path,
    artifact,
    weights_summary: Path,
    msm_summary: Path,
) -> Dict[str, Any]:
    entries: Dict[str, Any] = {
        "weights_summary": _relativize(weights_summary, output_dir),
        "msm_summary": _relativize(msm_summary, output_dir),
    }
    analysis_debug_dir = output_dir / "analysis_debug"
    if analysis_debug_dir.exists():
        entries["analysis_debug"] = _relativize(analysis_debug_dir, output_dir)
        summary_path = analysis_debug_dir / "summary.json"
        if summary_path.exists():
            entries["analysis_debug_summary"] = _relativize(summary_path, output_dir)

    if artifact and getattr(artifact, "bundle_path", None):
        entries["bundle"] = _relativize(Path(artifact.bundle_path), output_dir)
    if artifact and getattr(artifact, "debug_dir", None):
        entries["raw_debug_dir"] = str(Path(artifact.debug_dir))

    fes_candidate = output_dir / "fes_Tref.png"
    if fes_candidate.exists():
        entries["fes_plot"] = _relativize(fes_candidate, output_dir)
    return entries


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluate_guardrails(
    bundle: ExperimentBundle,
    artifact: Any,
) -> Tuple[list[Dict[str, Any]], Dict[str, float]]:
    guard_cfg = bundle.configs.msm.get("guardrails", {}) if bundle.configs.msm else {}
    summary = artifact.debug_summary or {}

    largest_scc = _coerce_float(
        summary.get("analysis_largest_scc_fraction")
        or summary.get("largest_scc_frame_fraction")
    )
    diag_mass = _coerce_float(summary.get("analysis_diag_mass") or summary.get("diag_mass"))

    counts_shape = summary.get("counts_shape") or []
    total_states = int(counts_shape[0]) if counts_shape else 0
    observed_states = int(summary.get("states_observed", total_states))
    empty_fraction = (
        float(total_states - observed_states) / float(total_states)
        if total_states > 0
        else 0.0
    )

    zero_rows = int(summary.get("analysis_zero_rows", summary.get("zero_rows", 0)))
    total_pairs = int(summary.get("analysis_total_pairs", summary.get("total_pairs", 0)))

    metrics = {
        "largest_scc_fraction": largest_scc,
        "empty_fraction": empty_fraction,
        "diag_mass": diag_mass,
        "zero_rows": float(zero_rows),
        "total_pairs": float(total_pairs),
    }

    violations: list[Dict[str, Any]] = []
    min_scc = guard_cfg.get("min_scc_fraction")
    if min_scc is not None and largest_scc is not None and largest_scc < float(min_scc):
        violations.append(
            {
                "code": "largest_scc_fraction_below_min",
                "threshold": float(min_scc),
                "actual": largest_scc,
            }
        )
    max_empty = guard_cfg.get("max_empty_fraction")
    if max_empty is not None and empty_fraction > float(max_empty):
        violations.append(
            {
                "code": "empty_fraction_above_max",
                "threshold": float(max_empty),
                "actual": empty_fraction,
            }
        )
    max_diag = guard_cfg.get("max_diagonal_mass")
    if (
        max_diag is not None
        and diag_mass is not None
        and diag_mass > float(max_diag)
    ):
        violations.append(
            {
                "code": "diag_mass_above_max",
                "threshold": float(max_diag),
                "actual": diag_mass,
            }
        )
    min_pairs = guard_cfg.get("min_total_pairs")
    if min_pairs is not None and total_pairs < float(min_pairs):
        violations.append(
            {
                "code": "total_pairs_below_min",
                "threshold": float(min_pairs),
                "actual": total_pairs,
            }
        )
    if guard_cfg.get("zero_rows_disallowed", False) and zero_rows:
        violations.append(
            {
                "code": "zero_row_states_present",
                "actual": zero_rows,
            }
        )
    return violations, metrics


def _sanitize_deeptica_payload(raw: Mapping[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    fields = [
        "applied",
        "skipped",
        "reason",
        "method",
        "lag",
        "lag_used",
        "n_out",
        "pairs_total",
        "warnings",
        "lag_candidates",
        "lag_fallback",
    ]
    for field in fields:
        if field in raw:
            summary[field] = raw[field]

    attempts = raw.get("attempts")
    if isinstance(attempts, list):
        trimmed: list[Dict[str, Any]] = []
        for attempt in attempts[:5]:
            if not isinstance(attempt, Mapping):
                continue
            trimmed.append(
                {
                    "lag": attempt.get("lag"),
                    "pairs_total": attempt.get("pairs_total"),
                    "status": attempt.get("status"),
                    "warnings": attempt.get("warnings"),
                }
            )
        if trimmed:
            summary["attempts"] = trimmed

    per_shard = raw.get("per_shard")
    if isinstance(per_shard, list):
        trimmed_shards: list[Dict[str, Any]] = []
        for shard_info in per_shard:
            if not isinstance(shard_info, Mapping):
                continue
            trimmed_shards.append(
                {
                    "shard_id": shard_info.get("shard_id") or shard_info.get("id"),
                    "pairs": shard_info.get("pairs"),
                    "frames": shard_info.get("frames"),
                }
            )
        if trimmed_shards:
            summary["per_shard"] = trimmed_shards

    training_metrics = raw.get("training_metrics")
    if isinstance(training_metrics, Mapping):
        summary["training_metrics"] = {
            "wall_time_s": training_metrics.get("wall_time_s"),
            "final_objective": training_metrics.get("final_objective"),
            "output_variance": training_metrics.get("output_variance"),
        }
    return summary


def _collect_deeptica_summary(
    bundle: ExperimentBundle,
    artifact: Any,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    transform_cfg = bundle.configs.transform or {}
    deeptica_cfg = (
        transform_cfg.get("deeptica")
        if isinstance(transform_cfg, Mapping)
        else {}
    )
    deeptica_enabled = bool(deeptica_cfg.get("enabled", False))
    acceptance_cfg = bundle.configs.msm.get("acceptance_checks", {}) if bundle.configs.msm else {}
    min_pairs = acceptance_cfg.get("deeptica_pairs_min")

    summary: Dict[str, Any] = {"status": "disabled"}
    violations: list[Dict[str, Any]] = []

    if not deeptica_enabled:
        return summary, violations

    summary["status"] = "missing"
    artifacts = getattr(artifact.build_result, "artifacts", {}) or {}
    raw_mlcv = artifacts.get("mlcv_deeptica")
    if not isinstance(raw_mlcv, Mapping):
        violations.append({"code": "deeptica_summary_missing"})
        return summary, violations

    sanitized = _sanitize_deeptica_payload(raw_mlcv)
    applied = bool(sanitized.get("applied"))
    skipped = bool(sanitized.get("skipped"))
    reason = sanitized.get("reason")
    pairs_total = sanitized.get("pairs_total")

    if applied:
        summary["status"] = "applied"
    elif skipped:
        summary["status"] = "skipped"
    else:
        summary["status"] = "unknown"

    summary.update(sanitized)

    if not applied:
        violations.append(
            {
                "code": "deeptica_not_applied",
                "status": summary["status"],
                "reason": reason,
            }
        )

    if min_pairs is not None and pairs_total is not None:
        try:
            threshold = float(min_pairs)
        except (TypeError, ValueError):
            threshold = None
        if threshold is not None and float(pairs_total) < threshold:
            violations.append(
                {
                    "code": "deeptica_pairs_below_min",
                    "threshold": threshold,
                    "actual": float(pairs_total),
                }
            )

    return summary, violations


def _build_msm_summary(bundle: ExperimentBundle, artifact: Any) -> Dict[str, Any]:
    if artifact is None:
        return {"status": "skipped", "reason": "analysis_not_executed"}
    br = artifact.build_result
    guardrail_violations, guardrail_metrics = _evaluate_guardrails(bundle, artifact)
    deeptica_summary, deeptica_violations = _collect_deeptica_summary(bundle, artifact)
    combined_violations = list(guardrail_violations)
    if artifact.guardrail_violations:
        combined_violations.extend(list(artifact.guardrail_violations or []))
    if deeptica_violations:
        combined_violations.extend(deeptica_violations)

    analysis_ok = artifact.analysis_healthy and not combined_violations
    return {
        "status": "ok" if analysis_ok else "fail",
        "guardrail_violations": _sanitize(combined_violations),
        "n_frames": int(getattr(br, "n_frames", 0)),
        "n_shards": int(getattr(br, "n_shards", 0)),
        "tau_frames": artifact.tau_frames,
        "effective_tau_frames": artifact.effective_tau_frames,
        "flags": _sanitize(getattr(br, "flags", {})),
        "messages": _sanitize(getattr(br, "messages", [])),
        "artifacts": _sanitize(getattr(br, "artifacts", {})),
        "metrics": _sanitize(guardrail_metrics),
        "deeptica": _sanitize(deeptica_summary),
        "deeptica_violations": _sanitize(deeptica_violations),
    }


def _generate_acceptance_report(
    output_dir: Path,
    experiment_name: str,
    weight_summary: WeightSummary,
    msm_summary: Mapping[str, Any],
    expected_failure: bool,
) -> tuple[Path, bool]:
    timestamp = datetime.now().isoformat()
    analysis_status = str(msm_summary.get("status", "skipped"))
    analysis_pass = analysis_status == "ok"

    if expected_failure:
        overall_pass = not weight_summary.passed or not analysis_pass
    else:
        overall_pass = weight_summary.passed and analysis_pass

    lines = [
        f"# {experiment_name} Acceptance Report",
        "",
        f"- Generated: {timestamp}",
        f"- Overall Status: {'PASS' if overall_pass else 'FAIL'}",
        f"- Expected Failure Scenario: {'Yes' if expected_failure else 'No'}",
        "",
        "## Weight Summary",
        f"- Status: {weight_summary.status.upper()}",
    ]
    global_ess = weight_summary.info.get("global_ess_fraction")
    if global_ess is not None:
        lines.append(f"- Global ESS Fraction: {global_ess:.4f}")
    violations = weight_summary.violations
    if violations:
        lines.append("- Violations:")
        for violation in violations:
            lines.append(f"  - {json.dumps(_sanitize(violation))}")
    else:
        lines.append("- Violations: None")

    lines.extend(
        [
            "",
            "## Analysis Summary",
            f"- Status: {analysis_status.upper()}",
        ]
    )
    analysis_violations = msm_summary.get("guardrail_violations") or []
    if analysis_violations:
        lines.append("- Guardrail Violations:")
        for violation in analysis_violations:
            lines.append(f"  - {json.dumps(_sanitize(violation))}")
    else:
        lines.append("- Guardrail Violations: None")

    deeptica_summary = msm_summary.get("deeptica") or {}
    deeptica_violations = msm_summary.get("deeptica_violations") or []
    if deeptica_summary:
        lines.extend(
            [
                "",
                "## DeepTICA Summary",
                f"- Status: {str(deeptica_summary.get('status', 'unknown')).upper()}",
                f"- Pairs Total: {deeptica_summary.get('pairs_total')}",
                f"- Reason: {deeptica_summary.get('reason')}",
                f"- Lag Used: {deeptica_summary.get('lag_used') or deeptica_summary.get('lag')}",
            ]
        )
    if deeptica_violations:
        lines.append("- DeepTICA Violations:")
        for violation in deeptica_violations:
            lines.append(f"  - {json.dumps(_sanitize(violation))}")
    elif deeptica_summary:
        lines.append("- DeepTICA Violations: None")

    artifact_paths = msm_summary.get("artifact_paths") or {}
    if artifact_paths:
        lines.extend(
            [
                "",
                "## Artifacts",
            ]
        )
        for name, path in sorted(artifact_paths.items()):
            lines.append(f"- {name}: {path}")

    report_path = output_dir / "acceptance_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path, overall_pass


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #


def _prepare_bundle(experiment_name: str, app_root: Path | None) -> ExperimentBundle:
    return load_bundle(experiment_name, app_root=app_root)


def _layout_app_path(app_root: Path | None) -> Path | None:
    if app_root is None:
        return None
    app_root = app_root.resolve()
    if app_root.is_dir():
        candidate = app_root / "__init__.py"
        if candidate.exists():
            return candidate
    return app_root


def run_experiment(
    experiment_name: str,
    *,
    app_root: Path | None = None,
    workspace: Path | None = None,
) -> int:
    bundle = _prepare_bundle(experiment_name, app_root)
    bundle.ensure_output_dir()
    output_dir = bundle.output_dir
    shards = bundle.load_shards()

    try:
        weight_summary = _compute_weights_summary(bundle, shards, bundle.configs.reweighter)
    except Exception as exc:
        LOGGER.debug("Weight summary computation failed", exc_info=True)
        violations = [{"code": "weights_summary_exception", "message": str(exc)}]
        weight_summary = WeightSummary(
            status="error",
            info={
                "mode": _normalize_reweight_mode(bundle.configs.reweighter.get("mode")),
                "reference_temperature_K": bundle.manifest.get("reference_temperature_K"),
                "error": str(exc),
                "violations": violations,
            },
            violations=violations,
        )
    weights_path = _write_json(output_dir / "weights_summary.json", weight_summary.info)

    expected_failure = bool(
        bundle.configs.msm.get("acceptance_checks", {}).get("expect_failure", False)
    )

    analysis_artifact = None
    analysis_status = "skipped"
    if weight_summary.passed or not expected_failure:
        # Build analysis unless weights already fail and failure is expected.
        effective_workspace = workspace or output_dir
        layout = _make_layout(_layout_app_path(app_root), effective_workspace)
        backend = WorkflowBackend(layout)
        config = _build_config(bundle, shards)
        analysis_artifact = backend.build_analysis(bundle.shard_jsons, config)
        _collect_debug_outputs(analysis_artifact, output_dir)
        msm_summary = _build_msm_summary(bundle, analysis_artifact)
        msm_path = _write_json(output_dir / "msm_summary.json", msm_summary)
        analysis_status = msm_summary.get("status", "fail")
    else:
        # Produce placeholder MSM summary since analysis was skipped intentionally.
        msm_summary = {
            "status": "skipped",
            "reason": "weights_guardrails_failed_before_analysis",
            "guardrail_violations": weight_summary.violations,
            "deeptica": {"status": "analysis_skipped"},
            "deeptica_violations": [],
        }
        msm_path = _write_json(output_dir / "msm_summary.json", msm_summary)
        placeholder_debug = {
            "status": "skipped",
            "reason": "Analysis not executed because weight guardrails failed.",
        }
        _write_json(output_dir / "analysis_debug" / "summary.json", placeholder_debug)

    artifact_paths = _collect_artifact_paths(
        output_dir=output_dir,
        artifact=analysis_artifact,
        weights_summary=weights_path,
        msm_summary=msm_path,
    )
    msm_summary["artifact_paths"] = _sanitize(artifact_paths)
    _write_json(msm_path, msm_summary)

    report_path, overall_pass = _generate_acceptance_report(
        output_dir,
        experiment_name,
        weight_summary,
        msm_summary,
        expected_failure,
    )

    status_text = "PASS" if overall_pass else "FAIL"
    deeptica_status = (
        (msm_summary.get("deeptica") or {}).get("status", "n/a")
        if isinstance(msm_summary, Mapping)
        else "n/a"
    )
    print(
        f"[{experiment_name}] overall status: {status_text} "
        f"(weights={weight_summary.status}, analysis={analysis_status}, deeptica={deeptica_status}). "
        f"Report: {report_path}",
        file=sys.stdout,
    )
    return 0 if overall_pass else 1


def run_experiment_cli(experiment_name: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Run the {experiment_name} experiment workflow.",
    )
    parser.add_argument(
        "--app-root",
        type=Path,
        default=None,
        help="Path to the app/ directory (defaults to package relative path).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Override workspace directory (defaults to the experiment output directory).",
    )
    args = parser.parse_args(argv)
    app_root = args.app_root
    if app_root is None:
        app_root = Path(__file__).resolve().parents[1]
    workspace = args.workspace
    return run_experiment(
        experiment_name=experiment_name,
        app_root=app_root,
        workspace=workspace,
    )


__all__ = ["run_experiment_cli", "run_experiment"]
