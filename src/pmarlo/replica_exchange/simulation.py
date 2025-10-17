"""Replica-exchange simulation utilities with mandatory dependencies."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from ._simulation_full import (
    Simulation,
    build_transition_model,
    plot_DG,
    prepare_system,
    production_run,
    relative_energies,
)

logger = logging.getLogger(__name__)


def feature_extraction(
    trajectory_file: str,
    topology_file: str,
    *,
    random_state: int | None = None,
    n_states: int = 40,
    stride: int = 1,
    **cluster_kwargs: Any,
) -> np.ndarray:
    """Cluster trajectory frames into microstates using lightweight defaults.

    Parameters
    ----------
    trajectory_file:
        Path to the trajectory file (DCD).  Only Cartesian coordinates are used.
    topology_file:
        Matching topology file (PDB) describing the atoms in the trajectory.
    random_state:
        Seed forwarded to :func:`pmarlo.api.cluster_microstates` for deterministic
        clustering.  ``None`` keeps the backend default.
    n_states:
        Target number of microstates.  Defaults to 40 for backwards compatibility
        with earlier workflows.
    stride:
        Optional frame thinning factor when loading the trajectory.
    **cluster_kwargs:
        Additional keyword arguments forwarded verbatim to the clustering API.
    """

    import mdtraj as md

    from pmarlo import api

    stride_int = max(1, int(stride))
    logger.info(
        "Loading trajectory '%s' with topology '%s' (stride=%d)",
        trajectory_file,
        topology_file,
        stride_int,
    )
    traj = md.load(trajectory_file, top=topology_file, stride=stride_int)
    if traj.n_frames == 0:
        raise ValueError(
            "Loaded trajectory contains no frames; cannot extract features"
        )

    coords = traj.xyz.reshape(traj.n_frames, -1)
    cluster_args: Dict[str, Any] = {
        "method": cluster_kwargs.pop("method", "auto"),
        "n_states": cluster_kwargs.pop("n_states", n_states),
        "random_state": random_state,
    }
    cluster_args.update(cluster_kwargs)

    logger.info(
        "Clustering %d frames into %s states (method=%s)",
        coords.shape[0],
        cluster_args.get("n_states", n_states),
        cluster_args.get("method", "auto"),
    )
    labels = api.cluster_microstates(coords, **cluster_args)
    return np.asarray(labels)
