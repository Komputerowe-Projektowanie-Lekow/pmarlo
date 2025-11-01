from __future__ import annotations

import json
import logging
from pathlib import Path

import mdtraj as md
import numpy as np

from pmarlo.utils.path_utils import ensure_directory

logger = logging.getLogger("pmarlo")


def extract_last_frame_to_pdb(
    *,
    trajectory_file: str | Path,
    topology_pdb: str | Path,
    out_pdb: str | Path,
    jitter_sigma_A: float = 0.0,
) -> Path:
    """Extract the last frame from a trajectory and write as a PDB.

    Parameters
    ----------
    trajectory_file:
        Path to the input trajectory (e.g., DCD).
    topology_pdb:
        PDB topology defining atom order.
    out_pdb:
        Destination PDB path to write.
    jitter_sigma_A:
        Optional Gaussian noise sigma in Angstroms applied to positions.

    Returns
    -------
    Path
        The output PDB path.
    """
    logger.info(
        "[trajectory_utils] Extracting last frame from trajectory: %s",
        Path(trajectory_file).name,
    )

    traj = md.load(str(trajectory_file), top=str(topology_pdb))
    if traj.n_frames <= 0:
        raise ValueError("Trajectory has no frames to extract")

    logger.debug(
        "[trajectory_utils] Trajectory loaded: %d frames, %d atoms",
        traj.n_frames,
        traj.n_atoms,
    )

    last = traj[traj.n_frames - 1]
    if jitter_sigma_A and float(jitter_sigma_A) > 0.0:
        logger.debug(
            "[trajectory_utils] Applying Gaussian jitter: sigma=%.2f Å",
            jitter_sigma_A,
        )
        noise = np.random.normal(0.0, float(jitter_sigma_A), size=last.xyz.shape)
        # MDTraj units are nm; 1 Å = 0.1 nm
        last.xyz = last.xyz + (noise * 0.1)

    out_p = Path(out_pdb)
    ensure_directory(out_p.parent)
    last.save_pdb(str(out_p))

    logger.info("[trajectory_utils] Last frame saved to: %s", out_p)
    return out_p


def extract_random_highT_frame_to_pdb(
    *,
    run_dir: str | Path,
    topology_pdb: str | Path,
    out_pdb: str | Path,
    jitter_sigma_A: float = 0.0,
    rng_seed: int | None = None,
) -> Path:
    """Extract a random frame from the highest-temperature replica of a run.

    Falls back to the last `replica_*.dcd` when metadata is missing.
    """
    logger.info(
        "[trajectory_utils] Extracting random frame from highest-T replica: run_dir=%s",
        Path(run_dir).name,
    )

    rd = Path(run_dir)
    analysis_json = rd / "replica_exchange" / "analysis_results.json"
    traj_path: Path | None = None

    if analysis_json.exists():
        logger.debug("[trajectory_utils] Found analysis results, parsing metadata")
        try:
            data = json.loads(analysis_json.read_text())
            remd = data.get("remd", {})
            temps = remd.get("temperatures", [])
            tfiles = remd.get("trajectory_files", [])
            if temps and tfiles and len(temps) == len(tfiles):
                # Choose highest temperature index
                # temps may be nested list from metadata-only; coerce
                temps_f = [float(x) for x in temps]
                i_max = int(np.argmax(temps_f))
                cand = Path(tfiles[i_max])
                traj_path = cand if cand.is_absolute() else (rd / cand)
                logger.info(
                    "[trajectory_utils] Selected highest-T replica: T=%.1fK, file=%s",
                    temps_f[i_max],
                    traj_path.name,
                )
        except Exception as e:
            logger.debug(
                "[trajectory_utils] Failed to parse analysis results: %s",
                str(e)[:100],
            )
            traj_path = None

    if traj_path is None:
        # No fallback - raise explicit error instead
        raise FileNotFoundError(
            f"No trajectory file found in analysis results at {rd}. "
            f"Expected analysis results to contain trajectory file paths or "
            f"replica_exchange/replica_*.dcd files to exist."
        )

    traj = md.load(str(traj_path), top=str(topology_pdb))
    if traj.n_frames <= 0:
        raise ValueError("Trajectory has no frames to extract")

    logger.debug(
        "[trajectory_utils] Trajectory loaded: %d frames, %d atoms",
        traj.n_frames,
        traj.n_atoms,
    )

    rng = np.random.default_rng(rng_seed)
    idx = int(rng.integers(0, traj.n_frames))
    logger.debug("[trajectory_utils] Selected random frame index: %d", idx)

    frame = traj[idx]
    if jitter_sigma_A and float(jitter_sigma_A) > 0.0:
        logger.debug(
            "[trajectory_utils] Applying Gaussian jitter: sigma=%.2f Å",
            jitter_sigma_A,
        )
        noise = np.random.normal(0.0, float(jitter_sigma_A), size=frame.xyz.shape)
        frame.xyz = frame.xyz + (noise * 0.1)

    out_p = Path(out_pdb)
    ensure_directory(out_p.parent)
    frame.save_pdb(str(out_p))

    logger.info("[trajectory_utils] Random frame saved to: %s", out_p)
    return out_p

