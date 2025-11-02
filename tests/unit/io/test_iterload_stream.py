import logging
from pathlib import Path

import pmarlo.io as io_module
from pmarlo.io import trajectory
from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM


def test_iterload_streaming(caplog):
    traj = Path("tests/_assets/traj.dcd")
    pdb = Path("tests/_assets/3gd8-fixed.pdb")
    msm = EnhancedMSM([str(traj)], topology_file=str(pdb))
    with caplog.at_level(logging.INFO):
        msm.load_trajectories(stride=2, atom_selection="name CA", chunk_size=5)
    assert msm.trajectories and msm.trajectories[0].n_frames == 50
    assert msm.trajectories[0].n_atoms < 500  # reduced atom count
    assert any("Streaming trajectory" in rec.getMessage() for rec in caplog.records)


def test_suppress_plugin_output_respects_runtime_toggle():
    """Ensure verbose flag changes are observed without re-importing."""

    logger_name = "mdtraj.formats.registry"
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    original_flag = io_module.verbose_plugin_logs

    try:
        io_module.verbose_plugin_logs = False
        logger.setLevel(logging.INFO)
        with trajectory._suppress_plugin_output():
            assert logging.getLogger(logger_name).level == logging.WARNING
        assert logging.getLogger(logger_name).level == logging.INFO

        io_module.verbose_plugin_logs = True
        logger.setLevel(logging.DEBUG)
        with trajectory._suppress_plugin_output():
            assert logging.getLogger(logger_name).level == logging.DEBUG
    finally:
        logger.setLevel(original_level)
        io_module.verbose_plugin_logs = original_flag
