from __future__ import annotations

from pathlib import Path

import numpy as np

from pmarlo.shards.format import write_shard
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta
from pmarlo.workflow.joint import JointWorkflow, WorkflowConfig


def _make_demo_shard(shard_id: str, temperature: float, n_frames: int) -> Shard:
    spec = FeatureSpec(name="demo", scaler="identity", columns=("x", "y"))
    beta = 1.0 / (0.00831446261815324 * temperature)
    meta = ShardMeta(
        schema_version="1.0",
        shard_id=shard_id,
        temperature_K=temperature,
        beta=beta,
        replica_id=0,
        segment_id=1,
        exchange_window_id=0,
        n_frames=n_frames,
        dt_ps=0.5,
        feature_spec=spec,
        provenance={"kind": "demux"},
    )
    X = np.column_stack(
        (
            np.linspace(-1.0, 1.0, n_frames, dtype=np.float32),
            np.sin(np.linspace(0.0, np.pi, n_frames, dtype=np.float32)),
        )
    )
    energy = np.linspace(0.0, 5.0, n_frames, dtype=np.float64)
    bias = np.zeros(n_frames, dtype=np.float64)
    return Shard(
        meta=meta,
        X=X,
        t_index=np.arange(n_frames, dtype=np.int64),
        dt_ps=meta.dt_ps,
        energy=energy,
        bias=bias,
    )


def test_joint_workflow_finalize_pipeline(tmp_path):
    shards_root = tmp_path / "shards"
    shards_root.mkdir()
    t_dir = shards_root / "T_300K"
    t_dir.mkdir()

    shard = _make_demo_shard("T300K_seg0001_rep000", 300.0, n_frames=12)
    write_shard(shard, t_dir)

    workflow = JointWorkflow(
        WorkflowConfig(
            shards_root=Path(shards_root),
            temperature_ref_K=300.0,
            tau_steps=1,
            n_clusters=2,
        )
    )

    result = workflow.finalize()

    assert result.T.shape == (2, 2)
    assert np.isclose(result.pi.sum(), 1.0)
    assert workflow.last_artifacts is not None
    assert "ck_errors" in workflow.last_artifacts
    assert result.meta.get("fes") is not None
    assert workflow.last_guardrails is not None
    assert "ck_transition_matrices" in workflow.last_artifacts
    assert "ck_row_counts" in workflow.last_artifacts
