# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
All Capabilities Demo (minimal): REMD → MSM → FES → Conformations

Outputs under: example_programs/programs_outputs/all_capabilities
Uses bundled test asset: tests/data/3gd8-fixed.pdb
"""

from __future__ import annotations

import logging
from pathlib import Path

from pmarlo import api, power_of_two_temperature_ladder

BASE_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BASE_DIR / "tests" / "data"
DEFAULT_PDB = TESTS_DIR / "3gd8-fixed.pdb"

OUT_DIR = Path(__file__).resolve().parent / "programs_outputs" / "all_capabilities"


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    pdb_path = DEFAULT_PDB.resolve()
    out_dir = ensure_output_dir(OUT_DIR)

    steps = 2000
    temperatures = power_of_two_temperature_ladder(300.0, 390.0, 8)

    def _print_progress(event: str, info):
        cur = info.get("current_step")
        tot = info.get("total_steps")
        eta = info.get("eta_s")
        msg = f"[{event}] elapsed={info.get('elapsed_s', 0):.1f}s"
        if cur is not None and tot is not None:
            msg += f" {int(cur)}/{int(tot)}"
        if eta is not None:
            msg += f" ETA={eta:.1f}s"
        print(msg)

    traj_files, analysis_temps = api.run_replica_exchange(
        pdb_file=pdb_path,
        output_dir=out_dir,
        temperatures=temperatures,
        total_steps=steps,
        progress_callback=_print_progress,
    )

    msm_dir = api.analyze_msm(
        trajectory_files=traj_files,
        topology_pdb=pdb_path,
        output_dir=out_dir,
        feature_type="universal_vamp",
        analysis_temperatures=analysis_temps,
    )

    api.find_conformations(
        topology_pdb=pdb_path,
        trajectory_choice=traj_files[0],
        output_dir=out_dir,
        feature_specs=["phi_psi", "chi1", "Rg", "sasa", "hbonds_count", "ssfrac"],
        requested_pair=None,
    )

    print(f"MSM analysis directory: {msm_dir}")
    print(f"Output base directory:  {out_dir}")
