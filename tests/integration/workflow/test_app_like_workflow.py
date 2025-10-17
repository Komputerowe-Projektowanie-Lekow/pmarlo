from __future__ import annotations

from pathlib import Path

import numpy as np

from pmarlo.utils.path_utils import ensure_directory

from pmarlo.data.aggregate import aggregate_and_build
from pmarlo.data.emit import emit_shards_from_trajectories
from pmarlo.data.shard import read_shard
from pmarlo.transform.build import AppliedOpts, BuildOpts, BuildResult
from pmarlo.transform.plan import TransformPlan, TransformStep


def _simple_extractor_factory(n_frames: int = 60):
    """Return an extractor callable matching emit_shards_from_trajectories.

    The extractor ignores the file contents and generates deterministic
    features based on the path name hash, to avoid heavy trajectory IO in tests.
    """

    def _extract(path: Path):
        rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
        t = np.linspace(0.0, 1.0, int(n_frames), endpoint=False)
        rg = (
            1.0
            + 0.1 * np.sin(2 * np.pi * t)
            + 0.01 * rng.standard_normal(int(n_frames))
        )
        rmsd = (
            0.5
            + 0.2 * np.cos(2 * np.pi * t)
            + 0.01 * rng.standard_normal(int(n_frames))
        )
        cvs = {"rg": rg.astype(np.float64), "rmsd": rmsd.astype(np.float64)}
        return cvs, None, {"source": str(path.name)}

    return _extract


def _alt_extractor_factory(n_frames: int = 60):
    """Variant extractor that changes CV names to trigger mismatch guardrails."""

    def _extract(path: Path):
        rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
        t = np.linspace(0.0, 1.0, int(n_frames), endpoint=False)
        rg = (
            1.0
            + 0.1 * np.sin(2 * np.pi * t)
            + 0.01 * rng.standard_normal(int(n_frames))
        )
        rmsd2 = (
            0.6
            + 0.1 * np.cos(2 * np.pi * t)
            + 0.01 * rng.standard_normal(int(n_frames))
        )
        cvs = {"rg": rg.astype(np.float64), "rmsd2": rmsd2.astype(np.float64)}
        return cvs, None, {"source": str(path.name)}

    return _extract


def _make_dummy_trajs(tmp: Path, n: int) -> list[Path]:
    paths: list[Path] = []
    for i in range(int(n)):
        p = tmp / f"traj_{i:02d}.dcd"
        ensure_directory(p.parent)
        p.write_bytes(b"")
        paths.append(p)
    return paths


def test_sharded_app_happy_path_emit_aggregate_build(tmp_path: Path):
    workspace = tmp_path / "app_workspace"
    shards_dir = workspace / "shards"
    bundles_dir = workspace / "bundles"
    ensure_directory(bundles_dir)

    trajs = _make_dummy_trajs(workspace / "trajs", 3)

    # Emit shards with deterministic synthetic extractor
    shard_jsons = emit_shards_from_trajectories(
        traj_files=trajs,
        out_dir=shards_dir,
        extract_cvs=_simple_extractor_factory(),
        temperature=300.0,
        periodic_by_cv={"rg": False, "rmsd": False},
    )
    assert len(shard_jsons) == 3

    # Validate shard metadata and payload shape
    meta, X, dtraj = read_shard(shard_jsons[0])
    assert tuple(meta.cv_names) == ("rg", "rmsd")
    assert tuple(meta.periodic) == (False, False)
    X = np.asarray(X)
    assert X.ndim == 2 and X.shape[1] == 2
    assert dtraj is None

    # Aggregate and build a bundle
    out_bundle = bundles_dir / "build.json"
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 0.6}),))
    opts = BuildOpts(seed=123, temperature=300.0, lag_candidates=[5, 10, 15])
    applied = AppliedOpts(
        bins={"rg": 24, "rmsd": 24}, lag=5, macrostates=5, notes={"app": "test"}
    )
    result, dataset_hash = aggregate_and_build(
        shard_jsons=shard_jsons,
        out_bundle=out_bundle,
        plan=plan,
        opts=opts,
        applied=applied,
    )

    assert out_bundle.exists(), "build bundle should be written"
    assert isinstance(result, BuildResult)
    assert isinstance(result.metadata.digest, str) and len(result.metadata.digest) > 0
    assert isinstance(dataset_hash, str) and len(dataset_hash) > 0
    assert result.flags.get("has_fes", False) is True, "FES expected from CVs"

    # Bundle round-trip load
    text = out_bundle.read_text()
    br = BuildResult.from_json(text)
    # FES payload is represented as FESResult or compatible dict
    assert br.fes is not None

    # Determinism: rebuilding with identical inputs yields same digest
    out_bundle2 = bundles_dir / "build2.json"
    result2, dataset_hash2 = aggregate_and_build(
        shard_jsons=shard_jsons,
        out_bundle=out_bundle2,
        plan=plan,
        opts=opts,
        applied=applied,
    )
    assert result2.metadata.digest == result.metadata.digest
    assert dataset_hash2 == dataset_hash


def test_sharded_app_mismatched_shards_raise(tmp_path: Path):
    workspace = tmp_path / "app_workspace"
    shards1 = workspace / "shards1"
    shards2 = workspace / "shards2"
    trajs1 = _make_dummy_trajs(workspace / "t1", 1)
    trajs2 = _make_dummy_trajs(workspace / "t2", 1)

    s1 = emit_shards_from_trajectories(
        traj_files=trajs1,
        out_dir=shards1,
        extract_cvs=_simple_extractor_factory(),
        temperature=300.0,
        periodic_by_cv={"rg": False, "rmsd": False},
    )
    s2 = emit_shards_from_trajectories(
        traj_files=trajs2,
        out_dir=shards2,
        extract_cvs=_alt_extractor_factory(),
        temperature=300.0,
        periodic_by_cv={"rg": False, "rmsd2": False},
    )

    out_bundle = workspace / "bundles" / "build.json"
    ensure_directory(out_bundle.parent)
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {}),))
    opts = BuildOpts(seed=0, temperature=300.0, lag_candidates=[5])
    applied = AppliedOpts(bins={"rg": 16, "rmsd": 16}, lag=5, macrostates=5)

    # Combining mismatched shards should raise a ValueError
    raised = False
    try:
        aggregate_and_build(
            shard_jsons=[*s1, *s2],
            out_bundle=out_bundle,
            plan=plan,
            opts=opts,
            applied=applied,
        )
    except ValueError as e:
        raised = True
        # Optional: check message hints at CV/periodicity mismatch
        assert "mismatch" in str(e).lower()
    assert raised, "Expected ValueError due to shard CV signature mismatch"
