from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta
from pmarlo_webapp.app.experiment.runner import (
    WeightSummary,
    _collect_artifact_paths,
    _collect_deeptica_summary,
    _compute_weights_summary,
    _generate_acceptance_report,
)


def _make_shard(shard_id: str, frames: int) -> Shard:
    feature_spec = FeatureSpec(
        name="test",
        scaler="unit",
        columns=("cv1", "cv2"),
    )
    meta = ShardMeta(
        schema_version="1.0",
        shard_id=shard_id,
        temperature_K=300.0,
        beta=0.4015,
        replica_id=0,
        segment_id=0,
        exchange_window_id=0,
        n_frames=frames,
        dt_ps=0.1,
        feature_spec=feature_spec,
        provenance={"seed": 1234},
    )
    X = np.zeros((frames, len(feature_spec.columns)), dtype=np.float32)
    t_index = np.arange(frames, dtype=np.int64)
    energy = np.linspace(0.0, 1.0, frames, dtype=np.float64)
    return Shard(
        meta=meta,
        X=X,
        t_index=t_index,
        dt_ps=meta.dt_ps,
        energy=energy,
    )


def test_compute_weights_summary_identity_uniform() -> None:
    shard_a = _make_shard("shard_000", 10)
    shard_b = _make_shard("shard_001", 15)
    bundle = SimpleNamespace(manifest={"reference_temperature_K": 300.0})

    summary = _compute_weights_summary(
        bundle=bundle,
        shards=[shard_a, shard_b],
        reweight_cfg={"mode": "identity"},
    )

    assert summary.status == "ok"
    assert summary.info["mode"] == "IDENTITY"
    assert pytest.approx(summary.info["global_ess_fraction"], rel=1e-6) == 1.0
    assert not summary.violations


def test_compute_weights_summary_guardrail_violation() -> None:
    shard = _make_shard("shard_000", 8)
    bundle = SimpleNamespace(manifest={"reference_temperature_K": 300.0})
    reweight_cfg = {
        "mode": "identity",
        "guardrails": {"min_effective_sample_fraction": 1.1},
    }

    summary = _compute_weights_summary(bundle, [shard], reweight_cfg)

    assert summary.status == "fail"
    assert summary.violations
    assert summary.violations[0]["code"] == "ess_fraction_below_threshold"


def test_collect_deeptica_summary_applied() -> None:
    deeptica_payload = {
        "applied": True,
        "pairs_total": 5000,
        "lag": 6,
        "per_shard": [{"shard_id": "shard_000", "pairs": 2500, "frames": 4000}],
        "warnings": [],
        "attempts": [{"lag": 6, "pairs_total": 5000, "status": "ok"}],
    }
    artifact = SimpleNamespace(
        build_result=SimpleNamespace(artifacts={"mlcv_deeptica": deeptica_payload})
    )
    bundle = SimpleNamespace(
        configs=SimpleNamespace(
            transform={"deeptica": {"enabled": True}},
            msm={"guardrails": {}, "acceptance_checks": {"deeptica_pairs_min": 4000}},
        )
    )

    summary, violations = _collect_deeptica_summary(bundle, artifact)

    assert summary["status"] == "applied"
    assert summary["pairs_total"] == 5000
    assert not violations


def test_collect_deeptica_summary_missing_artifact() -> None:
    artifact = SimpleNamespace(build_result=SimpleNamespace(artifacts={}))
    bundle = SimpleNamespace(
        configs=SimpleNamespace(
            transform={"deeptica": {"enabled": True}},
            msm={"guardrails": {}, "acceptance_checks": {}},
        )
    )

    summary, violations = _collect_deeptica_summary(bundle, artifact)

    assert summary["status"] == "missing"
    assert violations and violations[0]["code"] == "deeptica_summary_missing"


def test_generate_acceptance_report_includes_artifacts(tmp_path: Path) -> None:
    weight_summary = WeightSummary(
        status="ok",
        info={"global_ess_fraction": 0.95},
        violations=[],
    )
    artifact_paths = {
        "weights_summary": "weights_summary.json",
        "msm_summary": "msm_summary.json",
    }
    msm_summary = {
        "status": "ok",
        "guardrail_violations": [],
        "deeptica": {"status": "applied"},
        "deeptica_violations": [],
        "artifact_paths": artifact_paths,
    }

    report_path, overall_pass = _generate_acceptance_report(
        output_dir=tmp_path,
        experiment_name="E0_same_temp",
        weight_summary=weight_summary,
        msm_summary=msm_summary,
        expected_failure=False,
    )

    contents = report_path.read_text(encoding="utf-8")
    assert "## Artifacts" in contents
    assert "weights_summary.json" in contents
    assert overall_pass is True


def test_collect_artifact_paths_relativizes(tmp_path: Path) -> None:
    output_dir = tmp_path
    bundle_dir = output_dir / "bundle.pbz"
    bundle_dir.write_bytes(b"")
    weights_path = output_dir / "weights_summary.json"
    weights_path.write_text("{}", encoding="utf-8")
    msm_path = output_dir / "msm_summary.json"
    msm_path.write_text("{}", encoding="utf-8")
    analysis_dir = output_dir / "analysis_debug"
    analysis_dir.mkdir()
    (analysis_dir / "summary.json").write_text("{}", encoding="utf-8")

    artifact = SimpleNamespace(bundle_path=bundle_dir, debug_dir=analysis_dir)

    paths = _collect_artifact_paths(
        output_dir=output_dir,
        artifact=artifact,
        weights_summary=weights_path,
        msm_summary=msm_path,
    )

    assert paths["weights_summary"] == "weights_summary.json"
    assert paths["analysis_debug"] == "analysis_debug"
