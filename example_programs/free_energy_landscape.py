# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Free Energy Landscape example program using the public PMARLO API.

This script demonstrates an end-to-end workflow that:
- Loads a protein
- Runs a brief Replica Exchange Molecular Dynamics (REMD)
- Builds a Markov State Model (MSM)
- Generates and saves a 2D free energy surface (FES)

Outputs are written to:
  example_programs/programs_outputs/free_energy_landscape

Notes:
- The default inputs use the bundled test assets under `tests/data/`.
- To avoid requiring PDBFixer, we use the already fixed PDB (`3gd8-fixed.pdb`).
- The simulation and REMD here are intentionally short for demonstration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import mdtraj as md

# Import from the public package API (as if installed via pip)
from pmarlo import MarkovStateModel, Protein, ReplicaExchange

# ------------------------------ Configuration ------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TESTS_DIR = BASE_DIR / "tests" / "data"
DEFAULT_FIXED_PDB = DEFAULT_TESTS_DIR / "3gd8-fixed.pdb"

# Program outputs go here
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "programs_outputs" / "free_energy_landscape"
)


def configure_logging(verbose: bool) -> None:
    """Configure console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_output_dir(path: Path) -> Path:
    """Ensure the output directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_replica_exchange_simulation(
    pdb_file: Path,
    temperatures: List[float],
    steps: int,
    output_dir: Path,
) -> Tuple[List[str], List[float]]:
    """Run a brief REMD simulation and return list of trajectory files and temperatures.

    Returns a list of trajectory file paths (one per replica) and the temperature ladder
    used for analysis.
    """
    logging.info("Running REMD with temperatures: %s", temperatures)
    remd_output_dir = output_dir / "replica_exchange"

    # Choose stride to target ~5000 frames per replica (best effort)
    dcd_stride = max(1, int(steps // 5000))
    exchange_frequency = 200 if steps >= 20000 else 100

    remd = ReplicaExchange(
        pdb_file=str(pdb_file),
        temperatures=temperatures,
        output_dir=str(remd_output_dir),
        exchange_frequency=exchange_frequency,
        auto_setup=False,
        dcd_stride=dcd_stride,
    )
    remd.setup_replicas()
    remd.run_simulation(
        total_steps=int(steps), equilibration_steps=min(steps // 10, 200)
    )

    # Try to demultiplex to the closest to 300 K for MSM convenience; fall back to raw replicas
    demuxed = remd.demux_trajectories(
        target_temperature=300.0, equilibration_steps=min(steps // 10, 200)
    )
    if demuxed:
        # Count frames in demuxed; if insufficient, prefer TRAM on all replicas
        try:
            traj = md.load(str(demuxed), top=str(pdb_file))
            if traj.n_frames >= 1000:
                logging.info("Using demultiplexed trajectory at ~300K: %s", demuxed)
                return [demuxed], [300.0]
            logging.info(
                "Demux yielded only %d frames (<1000). Switching to multi-replica TRAM analysis.",
                traj.n_frames,
            )
        except Exception:
            logging.info(
                "Could not load demuxed trajectory reliably; using multi-replica TRAM."
            )

    # Fallback: use per-replica trajectories
    traj_files = [str(f) for f in remd.trajectory_files]
    logging.info(
        "Demultiplexing unavailable; using %d replica trajectories.", len(traj_files)
    )
    return traj_files, temperatures


def run_msm_and_fes(
    trajectory_files: List[str],
    topology_pdb: Path,
    output_dir: Path,
    n_states: int,
    feature_type: str,
    analysis_temperatures: Optional[List[float]] = None,
) -> Path:
    """Run MSM analysis and generate a 2D free energy surface.

    Returns the directory where analysis results were saved.
    """
    logging.info("Building MSM and generating free energy surface ...")
    msm_output_dir = output_dir / "msm_analysis"

    msm = MarkovStateModel(
        trajectory_files=trajectory_files,
        topology_file=str(topology_pdb),
        temperatures=analysis_temperatures or [300.0],
        output_dir=str(msm_output_dir),
    )

    # Analysis workflow
    msm.load_trajectories()
    msm.compute_features(feature_type=feature_type)
    # Estimate total frame count to choose states and bins later
    total_frames = 0
    try:
        total_frames = sum(t.n_frames for t in msm.trajectories)
    except Exception:
        total_frames = 0

    # Adapt number of states to data volume
    adaptive_states = (
        max(5, min(50, total_frames // 50)) if total_frames > 0 else n_states
    )
    msm.cluster_features(n_clusters=int(adaptive_states))

    # Choose MSM method based on number of trajectories/temperatures
    method = (
        "tram"
        if analysis_temperatures
        and len(analysis_temperatures) > 1
        and len(trajectory_files) > 1
        else "standard"
    )

    # First pass: quick MSM to compute ITS and select lag
    candidate_lags = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
    msm.build_msm(lag_time=5, method=method)
    msm.compute_implied_timescales(lag_times=candidate_lags, n_timescales=3)

    # Select lag time from ITS plateau (simple heuristic: smallest average relative slope)
    chosen_lag = 10
    try:
        import numpy as np

        lags = np.array(msm.implied_timescales["lag_times"])  # type: ignore[index]
        its = np.array(msm.implied_timescales["timescales"])  # type: ignore[index]
        scores: List[float] = []
        for idx, lag in enumerate(lags):
            if idx == 0:
                scores.append(float("inf"))
                continue
            prev = its[idx - 1]
            cur = its[idx]
            mask = np.isfinite(prev) & np.isfinite(cur) & (np.abs(prev) > 0)
            if np.count_nonzero(mask) == 0:
                scores.append(float("inf"))
                continue
            rel = np.mean(np.abs((cur[mask] - prev[mask]) / prev[mask]))
            scores.append(float(rel))
        # Prefer moderate lags; search excluding the first few tiny lags
        start_idx = min(3, len(lags) - 1)
        if start_idx < len(lags):
            region = scores[start_idx:]
            min_idx = int(np.nanargmin(region)) + start_idx
            chosen_lag = int(lags[min_idx])
    except Exception:
        chosen_lag = 10

    # Rebuild MSM with chosen lag time
    msm.build_msm(lag_time=chosen_lag, method=method)

    # FES in dihedral space by default (phi, psi)
    # Choose bins based on available frames to avoid overly sparse histograms
    adaptive_bins = max(20, min(50, int((total_frames or 0) ** 0.5))) or 20
    msm.generate_free_energy_surface(
        cv1_name="phi", cv2_name="psi", bins=adaptive_bins, temperature=300.0
    )

    # Save plots and results
    msm.plot_free_energy_surface(save_file="free_energy_surface", interactive=False)
    msm.plot_implied_timescales(save_file="implied_timescales")
    msm.plot_free_energy_profile(save_file="free_energy_profile")
    msm.create_state_table()
    msm.extract_representative_structures(save_pdb=True)
    msm.save_analysis_results()

    logging.info("MSM analysis complete. Results in: %s", msm_output_dir)
    return msm_output_dir


if __name__ == "__main__":
    # Linear, non-CLI execution
    configure_logging(verbose=False)

    pdb_path = DEFAULT_FIXED_PDB.resolve()
    output_dir = ensure_output_dir(DEFAULT_OUTPUT_DIR)
    steps = 1000
    n_states = 50
    feature_type = "phi_psi"
    temperatures = [300.0, 310.0, 320.0]

    # Initialize Protein using the public API; avoid automatic preparation to skip PDBFixer
    logging.info("Initializing protein: %s", pdb_path)
    protein = Protein(str(pdb_path), ph=7.0, auto_prepare=False)
    logging.info("Protein loaded. Properties: %s", protein.get_properties())

    # Run REMD and obtain trajectory files for analysis
    traj_files, analysis_temps = run_replica_exchange_simulation(
        pdb_file=pdb_path,
        temperatures=temperatures,
        steps=steps,
        output_dir=output_dir,
    )

    # MSM + Free Energy Surface
    msm_dir = run_msm_and_fes(
        trajectory_files=traj_files,
        topology_pdb=pdb_path,
        output_dir=output_dir,
        n_states=n_states,
        feature_type=feature_type,
        analysis_temperatures=analysis_temps,
    )

    print("\n=== Free Energy Landscape generation completed ===")
    print(f"MSM analysis directory: {msm_dir}")
    print(f"Output base directory:  {output_dir}")
    print(
        "Saved files include: free_energy_surface.png, implied_timescales.png, free_energy_profile.png"
    )
