from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.skipif(__import__("importlib.util").util.find_spec("openmm") is None, reason="Requires OpenMM")
def test_diagnostics_json_created(test_fixed_pdb_file: Path, tmp_path: Path):
    from example_programs.app_usecase.app.backend import run_short_sim

    ws = tmp_path
    temps = [300.0, 320.0, 340.0]
    sim = run_short_sim(Path(test_fixed_pdb_file), ws, temps, steps=1000, quick=True)
    diag = Path(sim.run_dir) / "replica_exchange" / "exchange_diagnostics.json"
    assert diag.exists()
    txt = diag.read_text(encoding="utf-8")
    assert "acceptance_mean" in txt and "mean_abs_disp_per_10k_steps" in txt

