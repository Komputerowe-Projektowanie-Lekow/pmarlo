from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pmarlo import emit_shards_from_trajectories, read_shard


def _deterministic_extractor_factory():
    def _extract(p: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray | None, Dict]:
        stem = p.stem
        try:
            idx = int("".join([c for c in stem if c.isdigit()]) or 0)
        except ValueError:
            idx = 0
        n = 20
        phi = np.linspace(-np.pi, np.pi, n) + 0.01 * idx
        psi = np.sin(np.linspace(0, 2 * np.pi, n)) + 0.02 * idx
        cvs = {"phi": phi, "psi": psi}
        source = {
            "note": f"traj:{stem}",
            "created_at": "1970-01-01T00:00:00Z",
            "kind": "demux",
            "run_id": "emit-test",
            "segment_id": idx,
            "replica_id": 0,
            "exchange_window_id": 0,
        }
        return cvs, None, source

    return _extract


def test_emit_three_shards(tmp_path: Path):
    trajs = []
    for i in range(3):
        f = tmp_path / f"traj_{i}.dcd"
        f.write_bytes(b"")
        trajs.append(f)

    out_dir = tmp_path / "shards"
    json_paths = emit_shards_from_trajectories(
        traj_files=trajs,
        out_dir=out_dir,
        extract_cvs=_deterministic_extractor_factory(),
        seed_start=10,
        temperature=310.0,
        periodic_by_cv={"phi": True, "psi": False},
    )
    assert len(json_paths) == 3
    hashes = []
    for j, jp in enumerate(json_paths):
        details, X, dtraj = read_shard(jp)
        assert dtraj is None
        assert X.shape[1] == 2
        assert details.cv_names == ("phi", "psi")
        assert details.periodic == (True, False)
        assert details.source["seed"] == 10 + j
        hashes.append((X.copy(), dtraj))
    # arrays differ due to the idx perturbation
    assert len({tuple(arr[0].ravel()) for arr in hashes}) == 3


def test_faulty_extractor_raises(tmp_path: Path):
    f = tmp_path / "traj.dcd"
    f.write_bytes(b"")

    def bad_extractor(p: Path):
        return {"a": np.arange(5.0), "b": np.arange(6.0)}, None, {}

    import pytest

    with pytest.raises(ValueError):
        emit_shards_from_trajectories([f], tmp_path, extract_cvs=bad_extractor)


def _constant_features() -> Dict[str, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 8, endpoint=False)
    return {"phi": grid}


def _make_source(
    *,
    kind: str,
    run_id: str,
    segment_id: int,
    replica_id: int,
) -> Dict[str, object]:
    return {
        "created_at": "1970-01-01T00:00:00Z",
        "kind": kind,
        "run_id": run_id,
        "segment_id": segment_id,
        "replica_id": replica_id,
        "exchange_window_id": 0,
    }


def test_shard_ids_include_run_id_and_kind(tmp_path: Path):
    demux_traj = tmp_path / "run-foo" / "demux" / "traj_a.dcd"
    replica_traj = tmp_path / "run-bar" / "replicas" / "traj_b.dcd"
    demux_traj.parent.mkdir(parents=True, exist_ok=True)
    replica_traj.parent.mkdir(parents=True, exist_ok=True)
    demux_traj.write_bytes(b"")
    replica_traj.write_bytes(b"")

    def _extract(kind: str, run_id: str, segment: int, replica: int):
        def _impl(_: Path):
            return (
                _constant_features(),
                None,
                _make_source(
                    kind=kind,
                    run_id=run_id,
                    segment_id=segment,
                    replica_id=replica,
                ),
            )

        return _impl

    demux_json = emit_shards_from_trajectories(
        [demux_traj],
        tmp_path / "demux_shards",
        extract_cvs=_extract("demux", "run-foo", segment=7, replica=2),
        temperature=300.0,
    )
    replica_json = emit_shards_from_trajectories(
        [replica_traj],
        tmp_path / "replica_shards",
        extract_cvs=_extract("replica", "run-bar", segment=3, replica=10),
        temperature=310.0,
    )

    assert Path(demux_json[0]).stem == "T300K_run-foo_seg0007_rep002"
    assert Path(replica_json[0]).stem == "replica_T310K_run-bar_seg0003_rep010"
