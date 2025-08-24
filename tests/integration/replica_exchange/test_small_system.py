from pathlib import Path

import mdtraj as md

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


def test_acceptance_and_demux(tmp_path):
    pdb = _write_water_pdb(tmp_path)
    remd = ReplicaExchange(
        pdb_file=str(pdb),
        temperatures=[300.0, 400.0],
        output_dir=tmp_path,
        exchange_frequency=5,
        auto_setup=True,
        dcd_stride=1,
        random_seed=1,
    )
    remd.run_simulation(total_steps=40, equilibration_steps=0)
    stats = remd.get_exchange_statistics()
    assert stats["total_exchange_attempts"] > 0
    assert 0.0 <= stats["overall_acceptance_rate"] <= 1.0
    demux_path = remd.demux_trajectories(
        target_temperature=300.0, equilibration_steps=0
    )
    assert demux_path is not None
    demux_file = Path(demux_path)
    assert demux_file.exists()
    traj = md.load(str(demux_file), top=str(pdb))
    assert traj.n_frames > 0
