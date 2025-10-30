from __future__ import annotations

import numpy as np

from pmarlo.shards.format import write_shard
from pmarlo.shards.schema import FeatureSpec, Shard, ShardMeta
from pmarlo.workflow.joint import JointWorkflow, WorkflowConfig


def _make_demo_shard(shard_id: str, temperature: float, n_frames: int) -> Shard:
    spec = FeatureSpec(name="demo", scaler="identity", columns=("x", "y"))
    beta = 1.0 / (0.00831446261815324 * temperature)
    seg_token = shard_id.split("_")[1]
    rep_token = shard_id.split("_")[2]
    segment_id = int(seg_token.replace("seg", ""))
    replica_id = int(rep_token.replace("rep", ""))
    meta = ShardMeta(
        schema_version="1.0",
        shard_id=shard_id,
        temperature_K=temperature,
        beta=beta,
        replica_id=replica_id,
        segment_id=segment_id,
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
    return Shard(
        meta=meta,
        X=X,
        t_index=np.arange(n_frames, dtype=np.int64),
        dt_ps=meta.dt_ps,
        energy=np.zeros(n_frames, dtype=np.float64),
    )


def test_iteration_invokes_remd_callback(tmp_path):
    shards_root = tmp_path / "shards"
    t_dir = shards_root / "T_300K"
    t_dir.mkdir(parents=True)

    shard = _make_demo_shard("T300K_seg0001_rep000", 300.0, 8)
    write_shard(shard, t_dir)

    workflow = JointWorkflow(
        WorkflowConfig(
            shards_root=shards_root,
            temperature_ref_K=300.0,
            tau_steps=1,
            n_clusters=2,
            use_reweight=False,
        )
    )

    captured = {}

    def fake_remd(bias_hook, iteration_index):
        captured["iteration"] = iteration_index
        bias_vals = bias_hook(np.array([[0.0], [0.5]]))
        captured["bias_sample"] = bias_vals
        new_id = f"T300K_seg999{iteration_index}_rep000"
        new_shard = _make_demo_shard(new_id, 300.0, 6)
        path = write_shard(new_shard, t_dir)
        return [path]

    class IdentityCV:
        def transform(self, X):
            return np.asarray(X, dtype=np.float64)

    workflow.cv_model = IdentityCV()
    workflow.set_remd_callback(fake_remd)
    metrics = workflow.iteration(0)

    assert metrics.notes == "iter 0"
    assert captured["iteration"] == 0
    assert isinstance(captured["bias_sample"], np.ndarray)
    assert workflow.last_new_shards
    assert workflow.last_new_shards[0].name.startswith("T300K_seg9990")
