from __future__ import annotations

import numpy as np
import pytest

from pmarlo.markov_state_model.msm_builder import MSMResult
from pmarlo.workflow.joint import CKGuardrailError, JointWorkflow, WorkflowConfig


class _TrainerStub:
    def __init__(self, vamp_series):
        self.history = [{"vamp2": val} for val in vamp_series]


def _make_workflow(tmp_path):
    cfg = WorkflowConfig(
        shards_root=tmp_path,
        temperature_ref_K=300.0,
        tau_steps=1,
        n_clusters=2,
        use_reweight=False,
    )
    wf = JointWorkflow(cfg)
    transition = np.eye(2, dtype=float)
    wf.last_artifacts = {
        "transition_matrix": transition,
        "ck_errors": {2: 0.05, 3: 0.08},
        "ck_transition_matrices": {2: transition.copy(), 3: transition.copy()},
        "ck_row_counts": {2: np.full(2, 100.0), 3: np.full(2, 120.0)},
    }
    wf.last_result = MSMResult(
        T=np.eye(2, dtype=float),
        pi=np.array([0.6, 0.4], dtype=float),
        its=np.array([1.0, 1.5], dtype=float),
        clusters=np.array([0, 1]),
        meta={},
    )
    return wf


def test_guardrails_ok(tmp_path):
    wf = _make_workflow(tmp_path)
    wf.trainer = _TrainerStub([0.1, 0.12, 0.15])
    report = wf.evaluate_guardrails()
    assert report.vamp2_trend_ok
    assert report.its_plateau_ok
    assert report.ck_threshold_ok


def test_guardrails_failure(tmp_path):
    wf = _make_workflow(tmp_path)
    wf.trainer = _TrainerStub([0.2, 0.15, 0.1])
    wf.last_artifacts["ck_transition_matrices"] = {
        2: np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        3: np.eye(2, dtype=float),
    }
    wf.last_artifacts["ck_row_counts"] = {
        2: np.full(2, 200.0),
        3: np.full(2, 180.0),
    }
    wf.last_artifacts["ck_errors"] = {2: 0.4}
    wf.last_result = MSMResult(
        T=np.eye(2, dtype=float),
        pi=np.array([0.6, 0.4], dtype=float),
        its=np.array([1.0, 0.5], dtype=float),
        clusters=np.array([0, 1]),
        meta={},
    )
    with pytest.raises(CKGuardrailError):
        wf.evaluate_guardrails(ck_mode="absolute", ck_absolute=0.15, ck_k_steps=(2,))
    assert not wf.last_guardrails.vamp2_trend_ok
    assert not wf.last_guardrails.its_plateau_ok
    assert not wf.last_guardrails.ck_threshold_ok
