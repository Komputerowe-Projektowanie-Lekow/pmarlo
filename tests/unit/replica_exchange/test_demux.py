from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import numpy as np
import pytest

from pmarlo.api import demultiplex_run
from pmarlo.io.trajectory_reader import MDTrajReader
from pmarlo.utils.json_io import load_json_file
from pmarlo.utils.path_utils import ensure_directory


def _write_minimal_pdb(path: Path, n_atoms: int = 1) -> Path:
    ensure_directory(path.parent)
    lines: List[str] = []
    for i in range(n_atoms):
        # Simple CA-only atoms; PDB requires fixed-width fields
        lines.append(
            f"ATOM  {i+1:5d}  CA  ALA A{1:4d}   {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
        )
    lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines))
    return path


def _write_replica_dcd(path: Path, topology: Path, values: List[float]) -> Path:
    from pmarlo.io.trajectory_writer import MDTrajDCDWriter

    coords = np.zeros((len(values), 1, 3), dtype=np.float32)
    for i, v in enumerate(values):
        coords[i, 0, 0] = float(v)
    w = MDTrajDCDWriter(rewrite_threshold=100)
    w.open(str(path), topology_path=str(topology), overwrite=True)
    w.write_frames(coords)
    w.close()
    return path


def _read_x_coords(path: Path, topology: Path) -> List[float]:
    reader = MDTrajReader(topology_path=str(topology))
    xs: List[float] = []
    n = reader.probe_length(str(path))
    for frame in reader.iter_frames(str(path), start=0, stop=n, stride=1):
        xs.append(float(frame[0, 0]))
    return xs


def _write_exchange_csv(path: Path, rows: List[List[int]]) -> Path:
    ensure_directory(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["slice"] + [f"replica_for_T{i}" for i in range(len(rows[0]))]
        writer.writerow(header)
        for i, row in enumerate(rows):
            writer.writerow([i, *row])
    return path


def test_demux_plan_exactness_and_lengths(tmp_path: Path):
    topo = _write_minimal_pdb(tmp_path / "system.pdb")
    # 3 replicas, 6 slices each
    N = 6
    ladder = [300.0, 310.0, 320.0]
    # Coordinates encode replica_index*1000 + frame_index
    rep0 = _write_replica_dcd(
        tmp_path / "replica_00.dcd", topo, [0 + i for i in range(N)]
    )
    rep1 = _write_replica_dcd(
        tmp_path / "replica_01.dcd", topo, [1000 + i for i in range(N)]
    )
    rep2 = _write_replica_dcd(
        tmp_path / "replica_02.dcd", topo, [2000 + i for i in range(N)]
    )

    # Cyclic mapping across slices: [0,1,2], [1,2,0], [2,0,1], repeat
    rows = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1],
    ]
    xlog = _write_exchange_csv(tmp_path / "exchange.csv", rows)

    out_dir = tmp_path / "demux"
    jsons = demultiplex_run(
        run_id="run-TEST",
        replica_traj_paths=[rep0, rep1, rep2],
        exchange_log_path=xlog,
        topology_path=topo,
        ladder_K=ladder,
        dt_ps=2.0,
        out_dir=out_dir,
        fmt="dcd",
        chunk_size=2,
    )
    assert len(jsons) == 3

    # Check output sequences for T0 and T1
    out_T0 = out_dir / "demux_T300K.dcd"
    out_T1 = out_dir / "demux_T310K.dcd"
    xs_T0 = _read_x_coords(out_T0, topo)
    xs_T1 = _read_x_coords(out_T1, topo)
    assert len(xs_T0) == N and len(xs_T1) == N
    # Expected: T0 sees r0,r1,r2,r0,r1,r2 → values [0,1000,2000,1,1001,2001]
    assert xs_T0 == [0.0, 1000.0, 2000.0, 1.0, 1001.0, 2001.0]
    # T1 sees r1,r2,r0,r1,r2,r0 → [1000,2000,0,1001,2001,1]
    assert xs_T1 == [1000.0, 2000.0, 0.0, 1001.0, 2001.0, 1.0]

    # JSON contents and lengths
    for jp, Tk in zip(sorted(jsons), [300.0, 310.0, 320.0]):
        data = load_json_file(jp)
        assert data["schema_version"] == "2.0"
        assert data["kind"] == "demux"
        assert data["run_id"] == "run-TEST"
        assert float(data["temperature_K"]) == Tk
        assert int(data["n_frames"]) == N
        assert isinstance(data.get("segments"), list)
        # Segments cover the output frames
        dst_starts = [s["dst_frame_start"] for s in data["segments"]]
        assert 0 in dst_starts


def test_reproducibility_hashes(tmp_path: Path):
    topo = _write_minimal_pdb(tmp_path / "system.pdb")
    N = 4
    rep0 = _write_replica_dcd(
        tmp_path / "replica_00.dcd", topo, [0 + i for i in range(N)]
    )
    rep1 = _write_replica_dcd(
        tmp_path / "replica_01.dcd", topo, [1000 + i for i in range(N)]
    )
    ladder = [300.0, 310.0]
    rows = [
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0],
    ]
    xlog = _write_exchange_csv(tmp_path / "exchange.csv", rows)
    out1 = tmp_path / "d1"
    out2 = tmp_path / "d2"
    j1 = demultiplex_run(
        run_id="run-TEST",
        replica_traj_paths=[rep0, rep1],
        exchange_log_path=xlog,
        topology_path=topo,
        ladder_K=ladder,
        dt_ps=1.0,
        out_dir=out1,
    )
    j2 = demultiplex_run(
        run_id="run-TEST",
        replica_traj_paths=[rep0, rep1],
        exchange_log_path=xlog,
        topology_path=topo,
        ladder_K=ladder,
        dt_ps=1.0,
        out_dir=out2,
    )
    # Compare hashes in JSON manifests
    h1 = [load_json_file(p)["integrity"]["traj_sha256"] for p in j1]
    h2 = [load_json_file(p)["integrity"]["traj_sha256"] for p in j2]
    assert sorted(h1) == sorted(h2)


def test_negative_cases(tmp_path: Path):
    topo = _write_minimal_pdb(tmp_path / "system.pdb")
    # ladder size mismatch
    rep0 = _write_replica_dcd(tmp_path / "r0.dcd", topo, [0, 1, 2])
    rep1 = _write_replica_dcd(tmp_path / "r1.dcd", topo, [1000, 1001, 1002])
    with pytest.raises(ValueError):
        demultiplex_run(
            run_id="run-TEST",
            replica_traj_paths=[rep0, rep1],
            exchange_log_path=tmp_path / "missing.csv",
            topology_path=topo,
            ladder_K=[300.0],  # mismatch
            dt_ps=1.0,
            out_dir=tmp_path / "out",
        )

    # malformed exchange row (duplicate assignments)
    xlog = _write_exchange_csv(tmp_path / "bad_exchange.csv", [[0, 0]])
    with pytest.raises(ValueError):
        demultiplex_run(
            run_id="run-TEST",
            replica_traj_paths=[rep0, rep1],
            exchange_log_path=xlog,
            topology_path=topo,
            ladder_K=[300.0, 310.0],
            dt_ps=1.0,
            out_dir=tmp_path / "out2",
        )

    # inconsistent replica length vs slices (replica consumed more than length)
    # Construct exchange with 4 slices but r1 will be used 4 times while only has 3 frames
    bad_rows = [[1, 0], [1, 0], [1, 0], [1, 0]]
    bad_xlog = _write_exchange_csv(tmp_path / "long_exchange.csv", bad_rows)
    with pytest.raises(ValueError):
        demultiplex_run(
            run_id="run-TEST",
            replica_traj_paths=[rep0, rep1],
            exchange_log_path=bad_xlog,
            topology_path=topo,
            ladder_K=[300.0, 310.0],
            dt_ps=1.0,
            out_dir=tmp_path / "out3",
        )
