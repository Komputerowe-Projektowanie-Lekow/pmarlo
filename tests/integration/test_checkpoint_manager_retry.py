from datetime import datetime, timedelta

from pmarlo.manager.checkpoint_manager import CheckpointManager


def test_retry_logic_and_state_cleanup(tmp_path):
    cm = CheckpointManager(
        output_base_dir=str(tmp_path),
        pipeline_steps=["step1", "step2"],
        max_retries=2,
    )
    cm.setup_run_directory()

    cm.mark_step_started("step1")
    cm.mark_step_failed("step1", "boom")
    assert cm.get_next_step() == "step1"

    cm.mark_step_started("step1")
    cm.mark_step_completed("step1")
    assert cm.get_next_step() == "step2"

    cm.mark_step_started("step2")
    cm.mark_step_failed("step2", "boom")
    cm.mark_step_started("step2")
    cm.mark_step_failed("step2", "boom again")
    assert cm.get_next_step() is None


def test_timeout_detection(tmp_path):
    cm = CheckpointManager(
        output_base_dir=str(tmp_path),
        pipeline_steps=["timeout_step"],
    )
    cm.setup_run_directory()

    cm.mark_step_started("timeout_step")
    cm.step_start_times["timeout_step"] = datetime.now() - timedelta(seconds=5)
    assert cm.has_timed_out("timeout_step", 1)
