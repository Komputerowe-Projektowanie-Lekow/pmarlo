from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from pmarlo.data.aggregate import aggregate_and_build
from pmarlo.data.emit import emit_shards_from_trajectories
from pmarlo.transform.build import AppliedOpts, BuildOpts
from pmarlo.transform.plan import TransformPlan, TransformStep


def _extractor_with_traj_ref(n_frames: int = 20):
    def _extract(path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray | None, Dict]:
        # Simple deterministic features, include 'traj' in source info
        t = np.linspace(0.0, 1.0, int(n_frames), endpoint=False)
        a = np.sin(2 * np.pi * t).astype(np.float64)
        b = np.cos(2 * np.pi * t).astype(np.float64)
        return {"a": a, "b": b}, None, {"traj": str(path), "n_frames": int(n_frames)}

    return _extract


def _make_file(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def test_demux_only_single_temperature_pass(tmp_path: Path):
    # Create demux-like trajectory files
    t300a = _make_file(tmp_path / "run-20250101-000000" / "demux_T300K.dcd")
    t300b = _make_file(tmp_path / "run-20250101-000000" / "demux_T300K_part2.dcd")
    out_dir = tmp_path / "shards"
    jsons = emit_shards_from_trajectories(
        traj_files=[t300a, t300b],
        out_dir=out_dir,
        extract_cvs=_extractor_with_traj_ref(),
        temperature=300.0,
        periodic_by_cv={"a": False, "b": False},
    )
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 0.6}),))
    opts = BuildOpts(seed=1, temperature=300.0, lag_candidates=[2, 4])
    applied = AppliedOpts(bins={"a": 16, "b": 16}, lag=2, macrostates=5)
    # Should not raise
    aggregate_and_build(
        jsons,
        opts=opts,
        plan=plan,
        applied=applied,
        out_bundle=tmp_path / "bundle.json",
    )


def test_demux_filename_without_keyword_is_accepted(tmp_path: Path):
    t300 = _make_file(tmp_path / "run-20250101-000000" / "segment_T300_part0.dcd")
    out_dir = tmp_path / "shards"
    jsons = emit_shards_from_trajectories(
        traj_files=[t300],
        out_dir=out_dir,
        extract_cvs=_extractor_with_traj_ref(),
        temperature=300.0,
        periodic_by_cv={"a": False, "b": False},
    )
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {"sigma": 0.4}),))
    opts = BuildOpts(seed=3, temperature=300.0, lag_candidates=[2, 4])
    applied = AppliedOpts(bins={"a": 12, "b": 12}, lag=2, macrostates=4)

    # Should not raise despite the trajectory filename lacking a demux hint
    aggregate_and_build(
        jsons,
        opts=opts,
        plan=plan,
        applied=applied,
        out_bundle=tmp_path / "bundle_demuxless.json",
    )


def test_mixed_kinds_are_rejected(tmp_path: Path):
    dmx = _make_file(tmp_path / "run-20250101-000000" / "demux_T300K.dcd")
    rep = _make_file(tmp_path / "run-20250101-000000" / "replica_00.dcd")
    out_dir = tmp_path / "shards"
    s1 = emit_shards_from_trajectories(
        [dmx], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=300.0
    )
    s2 = emit_shards_from_trajectories(
        [rep], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=300.0
    )
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {}),))
    opts = BuildOpts(seed=0, temperature=300.0, lag_candidates=[2])
    applied = AppliedOpts(bins={"a": 8, "b": 8}, lag=2, macrostates=3)
    with pytest.raises(ValueError):
        aggregate_and_build(
            [*s1, *s2],
            opts=opts,
            plan=plan,
            applied=applied,
            out_bundle=tmp_path / "bundle.json",
        )


def test_multi_temperature_demux_is_rejected(tmp_path: Path):
    t300 = _make_file(tmp_path / "run-20250101-000000" / "demux_T300K.dcd")
    t350 = _make_file(tmp_path / "run-20250101-000000" / "demux_T350K.dcd")
    out_dir = tmp_path / "shards"
    s1 = emit_shards_from_trajectories(
        [t300], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=300.0
    )
    s2 = emit_shards_from_trajectories(
        [t350], out_dir, extract_cvs=_extractor_with_traj_ref(), temperature=350.0
    )
    plan = TransformPlan(steps=(TransformStep("SMOOTH_FES", {}),))
    opts = BuildOpts(seed=0, temperature=300.0, lag_candidates=[2])
    applied = AppliedOpts(bins={"a": 8, "b": 8}, lag=2, macrostates=3)
    with pytest.raises(ValueError):
        aggregate_and_build(
            [*s1, *s2],
            opts=opts,
            plan=plan,
            applied=applied,
            out_bundle=tmp_path / "bundle.json",
        )
