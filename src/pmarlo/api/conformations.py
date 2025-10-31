import mdtraj as md
import logging
import numpy as np

from pmarlo.utils.mdtraj import load_mdtraj_topology, resolve_atom_selection
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.reporting.plots import save_fes_contour, save_transition_matrix_heatmap
from pmarlo.reporting.export import write_conformations_csv_json
from pmarlo.io import trajectory as traj_io

from .msm import (
    build_msm_from_labels,
    compute_macrostates,
    macrostate_populations,
    macro_transition_matrix,
    macro_mfpt,
)

from .fes import generate_fes_and_pick_minima
from .features import compute_features, reduce_features
from .clustering import cluster_microstates


from typing import Any, Mapping, Tuple, Sequence, Optional, Dict, List
from pathlib import Path

logger = logging.getLogger("pmarlo")


def sanitize_label_for_filename(name: str) -> str:
    return name.replace(":", "-").replace(" ", "_")

def find_conformations(  # noqa: C901
    topology_pdb: str | Path,
    trajectory_choice: str | Path,
    output_dir: str | Path,
    feature_specs: Optional[List[str]] = None,
    requested_pair: Optional[Tuple[str, str]] = None,
    traj_stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
) -> Path:
    """Find MSM- and FES-based representative conformations.

    Parameters
    ----------
    topology_pdb:
        Topology file in PDB format.
    trajectory_choice:
        Trajectory file to analyze.
    output_dir:
        Directory where results are written.
    feature_specs:
        Feature specification strings.
    requested_pair:
        Optional pair of feature names for FES plotting.
    traj_stride:
        Stride for loading trajectory frames.
    atom_selection:
        MDTraj atom selection string or indices used when loading the
        trajectory.
    chunk_size:
        Frames per chunk when streaming the trajectory.

    Returns
    -------
    Path
        The output directory path.
    """

    out = Path(output_dir)

    atom_indices: Sequence[int] | None = None
    if atom_selection is not None:
        topo = load_mdtraj_topology(topology_pdb)
        atom_indices = resolve_atom_selection(topo, atom_selection)

    logger.info(
        "Streaming trajectory %s with stride=%d, chunk=%d%s",
        trajectory_choice,
        traj_stride,
        chunk_size,
        f", selection={atom_selection}" if atom_selection else "",
    )
    traj: md.Trajectory | None = None

    loaded_frames = 0
    for chunk in traj_io.iterload(
        str(trajectory_choice),
        top=str(topology_pdb),
        stride=traj_stride,
        atom_indices=atom_indices,
        chunk=chunk_size,
    ):
        traj = chunk if traj is None else traj.join(chunk)
        loaded_frames += int(chunk.n_frames)
        if loaded_frames % max(1, chunk_size) == 0:
            logger.info("[stream] Loaded %d frames so far...", loaded_frames)
    if traj is None:
        raise ValueError("No frames loaded from trajectory")

    specs = feature_specs if feature_specs is not None else ["phi_psi"]
    cache_dir = Path(str(out)) / "feature_cache"
    ensure_directory(cache_dir)
    X, cols, periodic = compute_features(
        traj, feature_specs=specs, cache_path=str(cache_dir)
    )
    Y = reduce_features(X, method="vamp", lag=10, n_components=3)
    labels = cluster_microstates(Y, method="minibatchkmeans", n_states=8)

    dtrajs = [labels]
    observed_states = int(np.max(labels)) + 1 if labels.size else 0
    T, pi = build_msm_from_labels(dtrajs, n_states=observed_states, lag=10)
    macrostates = compute_macrostates(T, n_macrostates=4)
    _ = save_transition_matrix_heatmap(T, str(out), name="transition_matrix.png")

    items: List[dict] = []
    if macrostates is not None:
        macro_of_micro = macrostates
        macro_per_frame = macro_of_micro[labels]
        pi_macro = macrostate_populations(pi, macro_of_micro)
        T_macro = macro_transition_matrix(T, pi, macro_of_micro)
        mfpt = macro_mfpt(T_macro)

        for macro_id in sorted(set(int(m) for m in macro_per_frame)):
            idxs = np.where(macro_per_frame == macro_id)[0]
            if idxs.size == 0:
                continue
            centroid = np.mean(Y[idxs], axis=0)
            deltas = np.linalg.norm(Y[idxs] - centroid, axis=1)
            best_local = int(idxs[int(np.argmin(deltas))])
            best_local = int(best_local % max(1, traj.n_frames))
            rep_path = out / f"macrostate_{macro_id:02d}_rep.pdb"
            traj[best_local].save_pdb(str(rep_path))
            items.append(
                {
                    "type": "MSM",
                    "macrostate": int(macro_id),
                    "representative_frame": int(best_local),
                    "population": (
                        float(pi_macro[macro_id])
                        if pi_macro.size > macro_id
                        else float("nan")
                    ),
                    "mfpt_to": {
                        str(int(j)): float(mfpt[int(macro_id), int(j)])
                        for j in range(mfpt.shape[1])
                    },
                    "rep_pdb": str(rep_path),
                }
            )

    adaptive_bins = max(30, min(80, int((getattr(traj, "n_frames", 0) or 1) ** 0.5)))
    fes_info = generate_fes_and_pick_minima(
        X,
        cols,
        periodic,
        requested_pair=requested_pair,
        bins=(adaptive_bins, adaptive_bins),
        temperature=300.0,
        smooth=True,
        min_count=1,
        kde_bw_deg=(20.0, 20.0),
        deltaF_kJmol=3.0,
    )
    names = fes_info["names"]
    fes = fes_info["fes"]
    minima = fes_info["minima"]
    fname = f"fes_{sanitize_label_for_filename(names[0])}_vs_{sanitize_label_for_filename(names[1])}.png"
    if fes is not None:
        _ = save_fes_contour(
            fes.F,
            fes.xedges,
            fes.yedges,
            names[0],
            names[1],
            str(out),
            fname,
            mask=fes.metadata.get("mask"),
        )

    for idx, entry in enumerate(minima.get("minima", [])):
        frames = entry.get("frames", [])
        if not frames:
            continue
        best_local = int(frames[0])
        rep_path = out / f"state_{idx:02d}_rep.pdb"
        traj[best_local].save_pdb(str(rep_path))
        items.append(
            {
                "type": "FES_MIN",
                "state": int(idx),
                "representative_frame": int(best_local),
                "num_frames": int(entry.get("num_frames", 0)),
                "pair": {"x": names[0], "y": names[1]},
                "rep_pdb": str(rep_path),
            }
        )

    write_conformations_csv_json(str(out), items)
    return out


def find_conformations_with_msm(
    topology_pdb: str | Path,
    trajectory_file: str | Path,
    output_dir: str | Path,
    feature_specs: Optional[List[str]] = None,
    requested_pair: Optional[Tuple[str, str]] = None,
    traj_stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
) -> Path:
    """One-line convenience wrapper to find representative conformations.

    This is a thin alias around :func:`find_conformations` to mirror the
    example program name and make the public API more discoverable.
    """
    return find_conformations(
        topology_pdb=topology_pdb,
        trajectory_choice=trajectory_file,
        output_dir=output_dir,
        feature_specs=feature_specs,
        requested_pair=requested_pair,
        traj_stride=traj_stride,
        atom_selection=atom_selection,
        chunk_size=chunk_size,
    )
