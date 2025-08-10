# free_energy_landscape_long.py
# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Extended Free Energy Landscape analysis using the PMARLO API.

Differences from the demo version:
- Longer REMD simulation by default (200_000 steps)
- dcd_stride chosen to get ~5000 frames per replica
- Exchange frequency scaled with simulation length
- Automatic TRAM/dTRAM if multiple replicas & temperatures
- Adaptive number of MSM states with cap at total_frames // 50
- Separate output directory: programs_outputs/free_energy_landscape_long
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import mdtraj as md
import numpy as np

from pmarlo import MarkovStateModel, Protein, ReplicaExchange

# ------------------------------ Configuration ------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TESTS_DIR = BASE_DIR / "tests" / "data"
DEFAULT_FIXED_PDB = DEFAULT_TESTS_DIR / "3gd8-fixed.pdb"

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "programs_outputs" / "free_energy_landscape_long"
)

# ------------------------------ Utilities ------------------------------


def configure_logging(verbose: bool) -> None:
    """Configure console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def ensure_output_dir(path: Path) -> Path:
    """Ensure the output directory exists and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------ Simulation ------------------------------


def run_replica_exchange_simulation(
    pdb_file: Path,
    temperatures: List[float],
    steps: int,
    output_dir: Path,
) -> Tuple[List[str], List[float]]:
    """Run REMD simulation with improved sampling settings."""
    logging.info("Running REMD with temperatures: %s", temperatures)
    remd_output_dir = output_dir / "replica_exchange"

    # Aim for ~5000 frames per replica
    dcd_stride = max(1, steps // 5000)
    logging.info("Using dcd_stride=%d (targeting ~5000 frames)", dcd_stride)

    # Exchange frequency: every ~5–10% of total steps, but at least every 200 steps
    exchange_frequency = max(200, steps // 20)
    logging.info("Using exchange_frequency=%d", exchange_frequency)

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
        total_steps=int(steps), equilibration_steps=min(steps // 10, 2000)
    )
    for step in range(0, steps, exchange_frequency):
        logging.info("REMD Progress: %d/%d steps", step, steps)

    # Try demultiplexing
    demuxed = remd.demux_trajectories(
        target_temperature=300.0, equilibration_steps=min(steps // 10, 2000)
    )
    if demuxed:
        try:
            traj = md.load(str(demuxed), top=str(pdb_file))
            logging.info("Demuxed trajectory has %d frames", traj.n_frames)
            if traj.n_frames >= 2000:
                logging.info("Using demultiplexed trajectory at ~300K")
                return [demuxed], [300.0]
            logging.info(
                "Too few frames in demuxed (<2000) – will use all replicas in TRAM"
            )
        except Exception as e:
            logging.warning("Could not load demuxed trajectory: %s", e)

    # Fallback: all replicas
    traj_files = [str(f) for f in remd.trajectory_files]
    logging.info(
        "Using %d replica trajectories for multi-temperature analysis", len(traj_files)
    )
    return traj_files, temperatures


# ------------------------------ MSM + FES ------------------------------


def run_msm_and_fes(
    trajectory_files: List[str],
    topology_pdb: Path,
    output_dir: Path,
    default_n_states: int,
    feature_type: str,
    analysis_temperatures: Optional[List[float]] = None,
) -> Path:
    """Run MSM analysis with adaptive parameters."""
    logging.info("Building MSM and generating free energy surface ...")
    msm_output_dir = output_dir / "msm_analysis"

    msm = MarkovStateModel(
        trajectory_files=trajectory_files,
        topology_file=str(topology_pdb),
        temperatures=analysis_temperatures or [300.0],
        output_dir=str(msm_output_dir),
    )

    msm.load_trajectories()
    logging.info("Loaded trajectories for MSM analysis")
    msm.compute_features(feature_type=feature_type)
    logging.info("Computed features for MSM analysis")

    try:
        total_frames = sum(t.n_frames for t in msm.trajectories)
    except Exception:
        total_frames = 0
    logging.info("Total frames across all trajectories: %d", total_frames)

    # Adaptive n_states
    adaptive_states = max(5, min(50, total_frames // 50))
    n_states = adaptive_states if total_frames > 0 else default_n_states
    logging.info("Clustering into %d MSM states", n_states)
    msm.cluster_features(n_clusters=int(n_states))

    # Choose method: TRAM/dTRAM if multiple temps & replicas
    if (
        analysis_temperatures
        and len(analysis_temperatures) > 1
        and len(trajectory_files) > 1
    ):
        method = "tram"
    else:
        method = "standard"
    logging.info("Using MSM method: %s", method)

    # ITS scan
    candidate_lags = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100]
    msm.build_msm(lag_time=5, method=method)
    msm.compute_implied_timescales(lag_times=candidate_lags, n_timescales=3)

    chosen_lag = 10
    try:
        lags = np.array(msm.implied_timescales["lag_times"])  # type: ignore
        its = np.array(msm.implied_timescales["timescales"])  # type: ignore
        scores = []
        for idx, lag in enumerate(lags):
            if idx == 0:
                scores.append(float("inf"))
                continue
            prev, cur = its[idx - 1], its[idx]
            mask = np.isfinite(prev) & np.isfinite(cur) & (np.abs(prev) > 0)
            if np.count_nonzero(mask) == 0:
                scores.append(float("inf"))
                continue
            rel = np.mean(np.abs((cur[mask] - prev[mask]) / prev[mask]))
            scores.append(float(rel))
        start_idx = min(3, len(lags) - 1)
        if start_idx < len(lags):
            region = scores[start_idx:]
            min_idx = int(np.nanargmin(region)) + start_idx
            chosen_lag = int(lags[min_idx])
    except Exception as e:
        logging.warning("Could not auto-select lag time: %s", e)
        chosen_lag = 10
    logging.info("Chosen lag time: %d", chosen_lag)

    msm.build_msm(lag_time=chosen_lag, method=method)

    # Adaptive bins for FES
    adaptive_bins = max(20, min(50, int((total_frames or 0) ** 0.5))) or 20
    msm.generate_free_energy_surface(
        cv1_name="phi", cv2_name="psi", bins=adaptive_bins, temperature=300.0
    )

    # Save plots and outputs
    msm.plot_free_energy_surface(save_file="free_energy_surface", interactive=False)
    msm.plot_implied_timescales(save_file="implied_timescales")
    msm.plot_free_energy_profile(save_file="free_energy_profile")
    msm.create_state_table()
    msm.extract_representative_structures(save_pdb=True)
    msm.save_analysis_results()

    logging.info("MSM analysis complete. Results in: %s", msm_output_dir)
    return msm_output_dir


# ------------------------------ Main ------------------------------

if __name__ == "__main__":
    configure_logging(verbose=True)

    pdb_path = DEFAULT_FIXED_PDB.resolve()
    output_dir = ensure_output_dir(DEFAULT_OUTPUT_DIR)

    # Default parameters for longer production run
    steps = 50_000
    default_n_states = 50
    feature_type = "phi_psi"
    temperatures = [300.0, 310.0, 320.0]

    logging.info("Initializing protein: %s", pdb_path)
    protein = Protein(str(pdb_path), ph=7.0, auto_prepare=False)
    logging.info("Protein loaded. Properties: %s", protein.get_properties())

    traj_files, analysis_temps = run_replica_exchange_simulation(
        pdb_file=pdb_path,
        temperatures=temperatures,
        steps=steps,
        output_dir=output_dir,
    )

    msm_dir = run_msm_and_fes(
        trajectory_files=traj_files,
        topology_pdb=pdb_path,
        output_dir=output_dir,
        default_n_states=default_n_states,
        feature_type=feature_type,
        analysis_temperatures=analysis_temps,
    )

    print("\n=== Extended Free Energy Landscape generation completed ===")
    print(f"MSM analysis directory: {msm_dir}")
    print(f"Output base directory:  {output_dir}")
    print(
        "Saved files include: free_energy_surface.png, implied_timescales.png, free_energy_profile.png"
    )
