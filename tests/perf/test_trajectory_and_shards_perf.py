"""Micro-benchmarks for trajectory I/O, topology parsing, and shard utilities."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.benchmark]

# Optional plugin
pytest.importorskip("pytest_benchmark", reason="pytest-benchmark not installed")

if not os.getenv("PMARLO_RUN_PERF"):
    pytest.skip(
        "perf tests disabled; set PMARLO_RUN_PERF=1 to run", allow_module_level=True
    )


@dataclass(frozen=True)
class SyntheticTrajectory:
    pdb_path: str
    dcd_path: str
    n_atoms: int
    n_frames: int


def _make_linear_trajectory(
    tmp_path: Path,
    *,
    n_frames: int = 400,
    n_atoms: int = 6,
) -> SyntheticTrajectory:
    """Write a simple linear trajectory (PDB + DCD) for reader/writer tests."""

    top = md.Topology()
    chain = top.add_chain()
    residue = top.add_residue("GLY", chain)
    for _ in range(n_atoms):
        top.add_atom("C", md.element.carbon, residue)

    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for frame in range(n_frames):
        xyz[frame, :, 0] = frame  # encode frame index along x-axis
        xyz[frame, :, 1] = np.arange(n_atoms)  # atom index along y-axis

    pdb_path = tmp_path / "traj_topology.pdb"
    md.Trajectory(xyz[:1], top).save_pdb(pdb_path)

    dcd_path = tmp_path / "traj_data.dcd"
    md.Trajectory(xyz, top).save_dcd(dcd_path)

    return SyntheticTrajectory(
        pdb_path=str(pdb_path),
        dcd_path=str(dcd_path),
        n_atoms=n_atoms,
        n_frames=n_frames,
    )


def _build_shard_inputs(
    *, n_frames: int, n_cvs: int, seed: int, temperature: float = 300.0
):
    """Prepare deterministic shard emission payload."""

    from pmarlo import constants as const
    from pmarlo.shards.id import canonical_shard_id
    from pmarlo.shards.schema import FeatureSpec, ShardMeta

    rng = np.random.default_rng(seed)
    cv_names = tuple(f"cv_{i}" for i in range(n_cvs))
    cvs_matrix = rng.standard_normal((n_frames, n_cvs)).astype(np.float32)
    cvs = {name: cvs_matrix[:, idx] for idx, name in enumerate(cv_names)}
    periodic = {name: bool(idx % 2) for idx, name in enumerate(cv_names)}

    source_payload = {
        "kind": "demux",
        "run_id": f"benchmark_run_{seed}",
        "replica_id": seed % 5,
        "segment_id": seed,
        "exchange_window_id": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    feature_spec = FeatureSpec(
        name="synthetic_benchmark",
        scaler="identity",
        columns=cv_names,
    )
    provisional_meta = ShardMeta(
        schema_version=const.SHARD_SCHEMA_VERSION,
        shard_id="placeholder",
        temperature_K=temperature,
        beta=float(1.0 / (const.BOLTZMANN_CONSTANT_KJ_PER_MOL * temperature)),
        replica_id=source_payload["replica_id"],
        segment_id=source_payload["segment_id"],
        exchange_window_id=source_payload["exchange_window_id"],
        n_frames=n_frames,
        dt_ps=1.0,
        feature_spec=feature_spec,
        provenance=source_payload,
    )
    shard_id = canonical_shard_id(provisional_meta)

    write_kwargs = dict(
        shard_id=shard_id,
        cvs=cvs,
        dtraj=np.arange(n_frames, dtype=np.int32),
        periodic=periodic,
        seed=seed,
        temperature=temperature,
        source=source_payload,
        compute_arrays_hash=True,
        dtype=np.float32,
    )
    return write_kwargs, cv_names, periodic, source_payload


def test_mdttraj_reader_windowed_stream(benchmark, tmp_path):
    """Benchmark streaming window extraction via :class:`MDTrajReader`."""

    from pmarlo.io.trajectory_reader import MDTrajReader

    traj = _make_linear_trajectory(tmp_path, n_frames=360, n_atoms=5)
    reader = MDTrajReader(topology_path=traj.pdb_path)
    start, stop, stride = 24, 240, 4
    expected_frames = (stop - start + stride - 1) // stride

    def _stream():
        count = 0
        last_sum = 0.0
        for frame in reader.iter_frames(
            traj.dcd_path, start=start, stop=stop, stride=stride
        ):
            count += 1
            last_sum = float(np.sum(frame))
        return count, last_sum

    count, last_sum = benchmark(_stream)
    assert count == expected_frames
    assert last_sum >= 0.0  # ensure frames were materialised


def test_mdttraj_reader_probe_length(benchmark, tmp_path):
    """Benchmark length probing for trajectories."""

    from pmarlo.io.trajectory_reader import MDTrajReader

    traj = _make_linear_trajectory(tmp_path, n_frames=480, n_atoms=4)
    reader = MDTrajReader(topology_path=traj.pdb_path)

    result = benchmark(reader.probe_length, traj.dcd_path)
    assert result == traj.n_frames


def test_mdttraj_writer_chunked_rewrite(benchmark, tmp_path):
    """Benchmark :class:`MDTrajDCDWriter` chunked rewrite path."""

    from pmarlo.io.trajectory_writer import MDTrajDCDWriter

    traj = _make_linear_trajectory(tmp_path, n_frames=120, n_atoms=8)
    rng = np.random.default_rng(1337)
    batches = [
        rng.standard_normal((40, traj.n_atoms, 3)).astype(np.float32) for _ in range(3)
    ]
    counter = itertools.count()

    def _write_once() -> Path:
        idx = next(counter)
        out_path = tmp_path / f"writer_chunk_{idx}.dcd"
        writer = MDTrajDCDWriter(rewrite_threshold=48)
        writer.open(str(out_path), traj.pdb_path, overwrite=True)
        try:
            for chunk in batches:
                writer.write_frames(chunk)
        finally:
            writer.close()
        return out_path

    out_path = benchmark(_write_once)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_protein_topology_parsing(benchmark, tmp_path):
    """Benchmark topology parsing with :class:`pmarlo.protein.protein.Protein`."""

    from pmarlo.protein.protein import Protein

    traj = _make_linear_trajectory(tmp_path, n_frames=8, n_atoms=7)

    def _load_protein():
        protein = Protein(traj.pdb_path, auto_prepare=False)
        return (
            protein.properties["num_atoms"],
            protein.properties["num_residues"],
            protein.properties["num_chains"],
        )

    num_atoms, num_residues, num_chains = benchmark(_load_protein)
    assert num_atoms == 7
    assert num_residues == 1
    assert num_chains == 1


def test_shard_emission_throughput(benchmark, tmp_path):
    """Benchmark shard emission with hash computation enabled."""

    from pmarlo.data.shard import write_shard
    from pmarlo.data.shard_io import load_shard_meta

    payload, cv_names, periodic, _ = _build_shard_inputs(
        n_frames=512, n_cvs=12, seed=2024
    )
    base_dir = tmp_path / "shard_writes"
    counter = itertools.count()

    def _write_shard():
        idx = next(counter)
        out_dir = base_dir / f"batch_{idx}"
        return write_shard(out_dir=out_dir, **payload)

    json_path = benchmark(_write_shard)
    meta = load_shard_meta(json_path)
    assert meta.n_frames == 512
    assert tuple(meta.cv_names) == cv_names
    assert tuple(meta.periodic) == tuple(periodic.values())


def test_shard_aggregation_dataset_coalescing(benchmark, tmp_path):
    """Benchmark aggregation of multiple shards into a contiguous dataset."""

    from pmarlo.data.aggregate import _aggregate_shard_contents
    from pmarlo.data.shard import write_shard

    shard_dir = tmp_path / "aggregate_shards"
    shard_specs: list[tuple[int, tuple[str, ...]]] = []
    shard_paths: list[Path] = []
    for i, n_frames in enumerate((160, 200, 240, 280), start=1):
        payload, cv_names, _, _ = _build_shard_inputs(
            n_frames=n_frames, n_cvs=10, seed=500 + i
        )
        shard_path = write_shard(out_dir=shard_dir, **payload)
        shard_paths.append(shard_path)
        shard_specs.append((n_frames, cv_names))

    expected_frames = sum(spec[0] for spec in shard_specs)
    expected_cvs = shard_specs[0][1]

    def _aggregate():
        aggregated = _aggregate_shard_contents(tuple(shard_paths))
        return aggregated.X_all.shape

    shape = benchmark(_aggregate)
    assert shape == (expected_frames, len(expected_cvs))


def test_pair_builder_dense_pairs(benchmark, tmp_path):
    """Benchmark dense pair construction within a shard."""

    from pmarlo.data.shard import write_shard
    from pmarlo.shards.format import read_shard as read_full_shard
    from pmarlo.shards.pair_builder import PairBuilder

    payload, _, _, _ = _build_shard_inputs(n_frames=360, n_cvs=6, seed=900)
    shard_path = write_shard(out_dir=tmp_path / "pair_shards", **payload)
    shard = read_full_shard(shard_path)
    builder = PairBuilder(tau_steps=7)

    def _build_pairs():
        return builder.make_pairs(shard)

    pairs = benchmark(_build_pairs)
    assert pairs.shape == (shard.X.shape[0] - builder.tau, 2)


def test_pair_builder_sparse_pairs(benchmark, tmp_path):
    """Benchmark pair construction when ``n_frames <= tau`` (empty output)."""

    from pmarlo.data.shard import write_shard
    from pmarlo.shards.format import read_shard as read_full_shard
    from pmarlo.shards.pair_builder import PairBuilder

    payload, _, _, _ = _build_shard_inputs(n_frames=5, n_cvs=3, seed=1337)
    shard_path = write_shard(out_dir=tmp_path / "pair_shards_small", **payload)
    shard = read_full_shard(shard_path)
    builder = PairBuilder(tau_steps=10)

    def _build_pairs():
        return builder.make_pairs(shard)

    pairs = benchmark(_build_pairs)
    assert pairs.shape == (0, 2)
