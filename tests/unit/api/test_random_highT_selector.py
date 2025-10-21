from __future__ import annotations

from pathlib import Path

import pytest

from pmarlo.utils.path_utils import ensure_directory


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("mdtraj") is None,
    reason="Requires mdtraj",
)
def test_random_highT_frame_differs_from_base(
    test_trajectory_file: Path, test_fixed_pdb_file: Path, tmp_path: Path
):
    import json

    import mdtraj as md  # type: ignore

    from pmarlo.api import extract_random_highT_frame_to_pdb

    # Create a fake run directory structure with analysis_results.json
    rd = tmp_path / "run"
    ensure_directory(rd / "replica_exchange")
    # Copy or reference existing DCD by relative path
    dcd_rel = Path("replica_exchange") / "replica_00.dcd"
    # Symlink or copy
    try:
        (rd / dcd_rel).symlink_to(test_trajectory_file)
    except Exception:
        # Fallback copy
        import shutil

        shutil.copyfile(test_trajectory_file, rd / dcd_rel)
    analysis = {
        "remd": {
            "temperatures": [300.0],
            "trajectory_files": [str(dcd_rel)],
        }
    }
    (rd / "replica_exchange" / "analysis_results.json").write_text(json.dumps(analysis))

    out_pdb = tmp_path / "rand_highT.pdb"
    p = extract_random_highT_frame_to_pdb(
        run_dir=str(rd),
        topology_pdb=str(test_fixed_pdb_file),
        out_pdb=str(out_pdb),
        jitter_sigma_A=0.0,
        rng_seed=42,
    )
    assert p.exists()
    # Compute RMSD vs base PDB (should be > 0.1 Å)
    base = md.load(str(test_fixed_pdb_file))
    frame = md.load(str(p), top=str(test_fixed_pdb_file))
    # Align CA to reduce rigid-body effects
    ca = base.topology.select("name CA")
    if ca.size:
        frame = frame.superpose(base, atom_indices=ca)
        rmsd = md.rmsd(frame, base, atom_indices=ca)[0] * 10.0  # nm->Å
    else:
        rmsd = md.rmsd(frame, base)[0] * 10.0
    assert rmsd > 0.1
