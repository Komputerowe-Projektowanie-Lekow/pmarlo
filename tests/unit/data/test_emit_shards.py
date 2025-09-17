from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pmarlo import emit_shards_from_trajectories, read_shard


def _deterministic_extractor_factory():
    def _extract(p: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray | None, Dict]:
        # Derive a small integer from filename to perturb arrays deterministically
        stem = p.stem
        try:
            idx = int("".join([c for c in stem if c.isdigit()]) or 0)
        except ValueError:
            idx = 0
        n = 20
        phi = np.linspace(-np.pi, np.pi, n) + 0.01 * idx
        psi = np.sin(np.linspace(0, 2 * np.pi, n)) + 0.02 * idx
        cvs = {"phi": phi, "psi": psi}
        return cvs, None, {"note": f"traj:{stem}", "created_at": "1970-01-01T00:00:00Z"}

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
        meta, X, dtraj = read_shard(jp)
        assert dtraj is None
        assert X.shape[1] == 2
        assert tuple(meta.cv_names) == ("phi", "psi")
        assert tuple(meta.periodic) == (True, False)
        assert meta.seed == 10 + j
        hashes.append(meta.arrays_hash)
    # arrays differ due to the idx perturbation
    assert len(set(hashes)) == 3


def test_faulty_extractor_raises(tmp_path: Path):
    f = tmp_path / "traj.dcd"
    f.write_bytes(b"")

    def bad_extractor(p: Path):
        return {"a": np.arange(5.0), "b": np.arange(6.0)}, None, {}

    import pytest

    with pytest.raises(ValueError):
        emit_shards_from_trajectories([f], tmp_path, extract_cvs=bad_extractor)
