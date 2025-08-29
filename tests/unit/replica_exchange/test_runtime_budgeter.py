import logging
from pathlib import Path

import mdtraj as md
import pytest

from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


def _write_water_pdb(tmpdir: Path) -> Path:
    pdb_content = (
        "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  O   HOH A   1       0.000   0.000   0.000  1.00  0.00           O\n"
        "ATOM      2  H1  HOH A   1       0.957   0.000   0.000  1.00  0.00           H\n"
        "ATOM      3  H2  HOH A   1      -0.239   0.927   0.000  1.00  0.00           H\n"
        "TER\nEND\n"
    )
    pdb_path = tmpdir / "water.pdb"
    pdb_path.write_text(pdb_content)
    return pdb_path


def test_runtime_plan_dry_run(tmp_path, caplog):
    pdb = _write_water_pdb(tmp_path)
    remd = ReplicaExchange(
        pdb_file=str(pdb),
        temperatures=[300.0, 310.0],
        output_dir=tmp_path,
        dcd_stride=1,
        auto_setup=False,
    )
    caplog.set_level(logging.INFO, logger="pmarlo")
    plan = remd.plan_runtime(
        walltime=0.5,
        throughput_estimator=lambda: 1000.0,
        transitions_per_state=10,
        dry_run=True,
    )
    assert "Runtime plan" in caplog.text
    assert plan["total_steps"] > 0


def test_runtime_plan_execution(tmp_path):
    pdb = _write_water_pdb(tmp_path)
    remd = ReplicaExchange(
        pdb_file=str(pdb),
        temperatures=[300.0, 310.0],
        output_dir=tmp_path,
        dcd_stride=1,
        auto_setup=False,
    )
    plan = remd.plan_runtime(
        walltime=0.5,
        throughput_estimator=lambda: 1000.0,
        transitions_per_state=10,
    )
    remd.setup_replicas()
    traj_file = tmp_path / "replica_00.dcd"
    if traj_file.exists():
        try:
            before = md.load(str(traj_file), top=str(pdb)).n_frames
        except OSError:
            before = 0
    else:
        before = 0
    remd.run_simulation(
        total_steps=plan["total_steps"],
        equilibration_steps=plan["equilibration_steps"],
    )
    traj = md.load(str(traj_file), top=str(pdb))
    added = traj.n_frames - before
    equil_frames = plan["equilibration_steps"] // plan["reporter_stride"] + 1
    production_frames = added - equil_frames
    assert abs(production_frames - plan["expected_frames"]) <= max(
        1, plan["expected_frames"] // 10
    )


def test_plan_reporter_stride_single_application(tmp_path):
    pdb = _write_water_pdb(tmp_path)
    remd = ReplicaExchange(
        pdb_file=str(pdb),
        temperatures=[300.0, 310.0],
        output_dir=tmp_path,
        dcd_stride=1,
        auto_setup=False,
    )
    remd.plan_reporter_stride(100, 10, target_frames=10)
    with pytest.raises(AssertionError):
        remd.plan_reporter_stride(100, 10, target_frames=10)


def test_setup_requires_planned_stride(tmp_path):
    pdb = _write_water_pdb(tmp_path)
    remd = ReplicaExchange(
        pdb_file=str(pdb),
        temperatures=[300.0, 310.0],
        output_dir=tmp_path,
        dcd_stride=1,
        auto_setup=False,
    )
    with pytest.raises(AssertionError):
        remd.setup_replicas()
    remd.plan_reporter_stride(100, 10, target_frames=10)
    remd.setup_replicas()
