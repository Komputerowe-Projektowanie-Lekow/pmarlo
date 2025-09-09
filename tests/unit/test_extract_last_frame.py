from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("mdtraj") is None,
    reason="Requires mdtraj",
)
def test_extract_last_frame_to_pdb_smoke(
    test_trajectory_file: Path, test_fixed_pdb_file: Path, tmp_path: Path
):
    from pmarlo.api import extract_last_frame_to_pdb

    out = tmp_path / "last.pdb"
    p = extract_last_frame_to_pdb(
        trajectory_file=str(test_trajectory_file),
        topology_pdb=str(test_fixed_pdb_file),
        out_pdb=str(out),
        jitter_sigma_A=0.02,
    )
    assert p.exists()
