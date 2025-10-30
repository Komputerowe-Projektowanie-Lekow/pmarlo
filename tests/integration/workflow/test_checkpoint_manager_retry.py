import pytest

from pmarlo.transform.plan import TransformPlan, TransformStep
from pmarlo.transform.runner import TransformManifest, apply_plan

pytestmark = pytest.mark.integration


def test_transform_manifest_retry_logic(tmp_path):
    """Test retry logic using the new transform manifest system."""
    manifest = TransformManifest(tmp_path)

    # Initialize a simple plan
    manifest.init_run(
        TransformPlan(
            steps=(
                TransformStep(name="PROTEIN_PREPARATION", params={}),
                TransformStep(name="MSM_BUILD", params={}),
            )
        )
    )

    # Test step failure and retry tracking
    manifest.mark_step_start(0, "PROTEIN_PREPARATION", {})
    manifest.mark_step_failed(0, "PROTEIN_PREPARATION", "boom")

    # Should be marked as failed
    assert len(manifest.data["failed_steps"]) == 1
    assert manifest.data["failed_steps"][0]["step_name"] == "PROTEIN_PREPARATION"
    assert manifest.data["failed_steps"][0]["error"] == "boom"

    # Retry the step - mark as started then completed
    manifest.mark_step_start(0, "PROTEIN_PREPARATION", {})
    manifest.mark_step_complete(0, "PROTEIN_PREPARATION")

    # Should be marked as completed and removed from failed
    assert len(manifest.data["completed_steps"]) == 1
    assert len(manifest.data["failed_steps"]) == 0
    assert manifest.data["completed_steps"][0]["step_name"] == "PROTEIN_PREPARATION"


def test_transform_runner_with_retry(tmp_path):
    """Test the transform runner retry functionality."""

    def failing_transform(context, **kwargs):
        # Fail the first time, succeed the second
        if not hasattr(failing_transform, "called"):
            failing_transform.called = True
            raise RuntimeError("Simulated failure")
        return context

    # Mock the apply_transform_plan to use our failing function
    import pmarlo.transform.apply as apply_module
    import pmarlo.transform.runner as runner_module

    original_apply = apply_module.apply_transform_plan
    original_runner_apply = runner_module.apply_transform_plan

    def mock_apply(dataset, plan):
        for step in plan.steps:
            if step.name == "PROTEIN_PREPARATION":
                dataset = failing_transform(dataset, **step.params)
        return dataset

    apply_module.apply_transform_plan = mock_apply
    runner_module.apply_transform_plan = mock_apply

    try:
        plan = TransformPlan(
            steps=(TransformStep(name="PROTEIN_PREPARATION", params={}),)
        )

        # Should succeed after retry
        result = apply_plan(
            plan=plan,
            data={"test": "data"},
            checkpoint_dir=str(tmp_path),
            max_retries=1,
        )

        assert result is not None

        # Check that the manifest shows completion
        manifest = TransformManifest(tmp_path)
        manifest.load()
        assert len(manifest.data["completed_steps"]) == 1

    finally:
        # Restore original function
        apply_module.apply_transform_plan = original_apply
        runner_module.apply_transform_plan = original_runner_apply


def test_transform_manifest_persistence(tmp_path):
    """Test that transform manifests persist correctly."""
    manifest = TransformManifest(tmp_path)

    plan = TransformPlan(
        steps=(TransformStep(name="PROTEIN_PREPARATION", params={"test": "value"}),)
    )

    manifest.init_run(plan, "test_run_123")

    # Load in a new instance to verify persistence
    manifest2 = TransformManifest(tmp_path)
    manifest2.load()

    assert manifest2.data["run_id"] == "test_run_123"
    assert manifest2.data["status"] == "running"
    assert len(manifest2.data["completed_steps"]) == 0
