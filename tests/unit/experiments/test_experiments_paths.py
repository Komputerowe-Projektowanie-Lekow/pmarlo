from pathlib import Path

from pmarlo.experiments.cli import _tests_data_dir
from pmarlo.experiments.suite import _tests_pdb, _tests_traj


def test_cli_tests_data_dir_points_to_repo_tests():
    data_dir = _tests_data_dir()
    assert (data_dir / "3gd8-fixed.pdb").exists()


def test_suite_helpers_resolve_test_data():
    assert Path(_tests_pdb()).exists()
    for p in _tests_traj():
        assert Path(p).exists()
