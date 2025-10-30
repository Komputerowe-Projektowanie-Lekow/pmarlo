from __future__ import annotations

"""Performance micro-benchmarks for the demux facade and streaming engine."""

import os
import tracemalloc
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark]

# Optional plugin
pytest_benchmark = pytest.importorskip(
    "pytest_benchmark", reason="pytest-benchmark not installed"
)

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


def _make_replicas(
    tmp_path: Path, n_replicas: int = 5, n_frames: int = 400, n_atoms: int = 3
):
    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("GLY", chain)
    for _ in range(n_atoms):
        top.add_atom("C", md.element.carbon, residue)
    # Common PDB topology from first frame
    pdb = tmp_path / "top.pdb"
    md.Trajectory(np.zeros((1, n_atoms, 3), dtype=np.float32), top).save_pdb(pdb)
    dcds: list[str] = []
    for r in range(n_replicas):
        xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        for i in range(n_frames):
            xyz[i, :, 0] = i  # frame index as x
            xyz[i, :, 1] = r  # replica id as y
        traj = md.Trajectory(xyz, top)
        dcd = tmp_path / f"replica_{r:02d}.dcd"
        traj.save_dcd(dcd)
        dcds.append(str(dcd))
    return str(pdb), dcds


def _build_remd(pdb: str, dcds: list[str], tmp_path: Path):
    from pmarlo.replica_exchange.replica_exchange import ReplicaExchange

    remd = ReplicaExchange.__new__(ReplicaExchange)
    remd.pdb_file = pdb
    remd.trajectory_files = [Path(p) for p in dcds]
    remd.temperatures = [300.0 + 10.0 * i for i in range(len(dcds))]
    remd.n_replicas = len(dcds)
    # 1 frame per segment, map fixed states so target is replica 0 always
    n_segments = 200
    remd.exchange_history = [[0] + list(range(1, len(dcds))) for _ in range(n_segments)]
    remd.reporter_stride = 1
    remd.dcd_stride = 1
    remd.exchange_frequency = 1
    remd.output_dir = tmp_path
    remd.integrators = []
    remd._replica_reporter_stride = [1 for _ in dcds]
    return remd


def _benchmark_memory(func, *args, **kwargs):
    tracemalloc.start()
    try:
        return func(*args, **kwargs), tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()


def test_perf_demux_facade(benchmark, tmp_path: Path):
    from pmarlo.demultiplexing.demux import demux_trajectories

    pdb, dcds = _make_replicas(tmp_path)
    remd = _build_remd(pdb, dcds, tmp_path)

    def _run():
        return demux_trajectories(remd, target_temperature=300.0, equilibration_steps=0)

    def _bench():
        path, (cur, peak) = _benchmark_memory(_run)
        return path, cur, peak

    result = benchmark(_bench)
    # Attach memory info to output for human inspection
    path, cur, peak = result
    print(f"demux facade: out={path} peak_mem_bytes={peak}")


def test_perf_streaming_demux(benchmark, tmp_path: Path):
    from pmarlo.demultiplexing.demux_engine import demux_streaming
    from pmarlo.demultiplexing.demux_plan import build_demux_plan
    from pmarlo.io.trajectory_reader import MDTrajReader
    from pmarlo.io.trajectory_writer import MDTrajDCDWriter

    pdb, dcds = _make_replicas(tmp_path)
    n_replicas = len(dcds)
    temperatures = [300.0 + 10.0 * i for i in range(n_replicas)]
    n_segments = 200
    exchange_history = [[0] + list(range(1, n_replicas)) for _ in range(n_segments)]

    reader = MDTrajReader(topology_path=pdb)
    frames = [reader.probe_length(p) for p in dcds]
    plan = build_demux_plan(
        exchange_history=exchange_history,
        temperatures=temperatures,
        target_temperature=300.0,
        exchange_frequency=1,
        equilibration_offset=0,
        replica_paths=dcds,
        replica_frames=frames,
        default_stride=1,
    )

    out = tmp_path / "demux_streaming.dcd"

    def _run():
        writer = MDTrajDCDWriter(rewrite_threshold=2048)
        writer.open(str(out), pdb, overwrite=True)
        try:
            return demux_streaming(plan, pdb, reader, writer, fill_policy="repeat")
        finally:
            writer.close()

    def _bench():
        res, (cur, peak) = _benchmark_memory(_run)
        return res.total_frames_written, cur, peak

    result = benchmark(_bench)
    total, cur, peak = result
    print(f"streaming demux: frames={total} peak_mem_bytes={peak}")
