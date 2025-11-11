from __future__ import annotations

from pathlib import Path

from pmarlo.api import build_joint_workflow
from pmarlo.workflow.joint import JointWorkflow, WorkflowConfig
from pmarlo.workflow.metrics import Metrics


def test_metrics_dataclass():
    metrics = Metrics(vamp2_val=0.1, its_val=0.2, ck_error=0.3, notes="stub")
    assert metrics.notes == "stub"
    assert metrics.vamp2_val == 0.1


def test_joint_workflow_instantiation(tmp_path):
    cfg = WorkflowConfig(
        shards_root=Path(tmp_path),
        temperature_ref_K=300.0,
        tau_steps=1,
        n_clusters=2,
        use_reweight=False,
    )
    workflow = JointWorkflow(cfg)
    workflow.bootstrap_cv()
    metrics = workflow.iteration(0)
    assert isinstance(metrics, Metrics)

    wf_from_api = build_joint_workflow(
        shards_root=tmp_path,
        temperature_ref_K=300.0,
        tau_steps=1,
        n_clusters=2,
        use_reweight=False,
    )
    assert isinstance(wf_from_api, JointWorkflow)
