# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Minimal Free Energy Landscape demo using PMARLO's public API.

This short example runs a brief replica exchange simulation,
builds a Markov state model and produces a 2D free energy surface.
All heavy lifting is performed by high level helpers in ``pmarlo.api``.
"""

from __future__ import annotations

from pathlib import Path

from pmarlo import Protein, power_of_two_temperature_ladder
from pmarlo.api import analyze_msm, run_replica_exchange


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    pdb = base / "tests" / "data" / "3gd8-fixed.pdb"
    out = Path(__file__).parent / "programs_outputs" / "free_energy_landscape"
    out.mkdir(parents=True, exist_ok=True)

    temps = power_of_two_temperature_ladder(300.0, 375.0, 16)
    Protein(str(pdb), ph=7.0, auto_prepare=False)

    trajs, analysis_temps = run_replica_exchange(
        pdb_file=pdb, output_dir=out, temperatures=temps, total_steps=1000
    )

    msm_dir = analyze_msm(
        trajectory_files=trajs,
        topology_pdb=pdb,
        output_dir=out,
        feature_type="phi_psi",
        analysis_temperatures=analysis_temps,
    )

    print("MSM analysis directory:", msm_dir)
    print("Output base directory:", out)
    print(
        "Saved files include: free_energy_surface.png, "
        "implied_timescales.png, free_energy_profile.png"
    )


if __name__ == "__main__":
    main()
