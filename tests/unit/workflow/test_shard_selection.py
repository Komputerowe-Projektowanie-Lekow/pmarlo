from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pmarlo.data.emit import emit_shards_from_trajectories
from pmarlo.transform.build import group_demux_shards_by_temperature, select_shards
from pmarlo.utils.path_utils import ensure_directory


def _extractor_with_traj_ref(n_frames: int = 10):
    def _extract(path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray | None, Dict]:
        # Simple deterministic features; include source traj for provenance
        t = np.linspace(0.0, 1.0, int(n_frames), endpoint=False)
        return {"u": t, "v": t**2}, None, {"traj": str(path), "n_frames": int(n_frames)}

    return _extract


def _mk_file(p: Path) -> Path:
    ensure_directory(p.parent)
    p.write_bytes(b"")
    return p


def test_select_shards_filters_demux_and_temperature(tmp_path: Path):
    # demux temps
    t300 = _mk_file(tmp_path / "run-20250101-000000" / "demux_T300K.dcd")
    t350 = _mk_file(tmp_path / "run-20250101-000000" / "demux_T350K.dcd")
    # replica file
    rep0 = _mk_file(tmp_path / "run-20250101-000000" / "replica_00.dcd")

    out_dir = tmp_path / "shards"
    s300 = emit_shards_from_trajectories(
        [t300], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=300.0
    )
    s350 = emit_shards_from_trajectories(
        [t350], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=350.0
    )
    srep = emit_shards_from_trajectories(
        [rep0], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=300.0
    )

    # Mixed list
    all_jsons = [*s300, *s350, *srep]

    # DEMUX-only selection drops replica shards
    demux_only = select_shards(all_jsons, mode="demux")
    assert set(map(lambda p: Path(p).name, demux_only)) == set(
        map(lambda p: Path(p).name, [*s300, *s350])
    )

    # Temperature filter keeps only nearby T
    demux_300 = select_shards(all_jsons, mode="demux", demux_temperature=300.0)
    names_300 = set(map(lambda p: Path(p).name, demux_300))
    assert names_300 == set(map(lambda p: Path(p).name, s300))


def test_group_demux_shards_by_temperature(tmp_path: Path):
    t300a = _mk_file(tmp_path / "run-20250101-000000" / "demux_T300K.dcd")
    t300b = _mk_file(tmp_path / "run-20250101-000000" / "demux_T300K_part2.dcd")
    t340 = _mk_file(tmp_path / "run-20250101-000000" / "demux_T340K.dcd")
    out_dir = tmp_path / "shards"
    s300a = emit_shards_from_trajectories(
        [t300a], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=300.0
    )
    s300b = emit_shards_from_trajectories(
        [t300b], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=300.0
    )
    s340 = emit_shards_from_trajectories(
        [t340], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=340.0
    )

    groups = group_demux_shards_by_temperature([*s300a, *s300b, *s340])
    assert set(groups.keys()) == {300.0, 340.0}
    assert set(map(lambda p: Path(p).name, groups[300.0])) == set(
        map(lambda p: Path(p).name, [*s300a, *s300b])
    )
    assert set(map(lambda p: Path(p).name, groups[340.0])) == set(
        map(lambda p: Path(p).name, s340)
    )
