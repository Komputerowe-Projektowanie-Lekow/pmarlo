from __future__ import annotations

from pathlib import Path

from pmarlo.workflow.joint import JointWorkflow, WorkflowConfig


def test_joint_workflow_import_and_config(tmp_path):
    cfg = WorkflowConfig(
        shards_root=Path(tmp_path),
        temperature_ref_K=300.0,
        tau_steps=1,
        n_clusters=2,
        use_reweight=False,
    )
    workflow = JointWorkflow(cfg)
    assert workflow.cfg.shards_root == Path(tmp_path)
