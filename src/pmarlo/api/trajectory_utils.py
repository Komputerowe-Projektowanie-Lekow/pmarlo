from __future__ import annotations

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


