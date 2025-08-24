from pathlib import Path

import matplotlib
import mdtraj as md
import numpy as np

from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM
from pmarlo.replica_exchange.demux_metadata import DemuxMetadata
from pmarlo.replica_exchange.replica_exchange import ReplicaExchange
from pmarlo.results import ITSResult

matplotlib.use("Agg")


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


def test_demux_metadata_roundtrip(tmp_path):
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
    remd.run_simulation(total_steps=90, equilibration_steps=0)
    n_segments = len(remd.exchange_history)
    assert n_segments == 18

    demux_path = remd.demux_trajectories(
        target_temperature=300.0, equilibration_steps=0
    )
    assert demux_path is not None
    demux_file = Path(demux_path)
    meta_file = demux_file.with_suffix(".meta.json")
    assert demux_file.exists() and meta_file.exists()

    meta = DemuxMetadata.from_json(meta_file)
    roundtrip = DemuxMetadata.from_dict(meta.to_dict())
    assert meta == roundtrip

    traj = md.load(str(demux_file), top=str(pdb))
    assert traj.n_frames == n_segments * meta.frames_per_segment

    msm = EnhancedMSM(
        trajectory_files=[str(demux_file)],
        topology_file=str(pdb),
        output_dir=str(tmp_path / "msm"),
    )
    msm.load_trajectories()
    assert np.isclose(
        msm.time_per_frame_ps,
        meta.integration_timestep_ps
        * meta.exchange_frequency_steps
        / meta.frames_per_segment,
    )

    msm.implied_timescales = ITSResult(
        lag_times=np.array([1]),
        eigenvalues=np.array([[0.5]]),
        eigenvalues_ci=np.array([[[0.4, 0.6]]]),
        timescales=np.array([[1.0]]),
        timescales_ci=np.array([[[0.8, 1.2]]]),
        rates=np.array([[1.0]]),
        rates_ci=np.array([[[0.8, 1.2]]]),
        recommended_lag_window=(msm.time_per_frame_ps, msm.time_per_frame_ps),
    )
    msm.plot_implied_timescales()
    xlabel = matplotlib.pyplot.gca().get_xlabel()
    assert "ps" in xlabel or "ns" in xlabel
