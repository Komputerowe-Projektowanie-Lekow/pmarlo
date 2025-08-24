"""Simplified example pipeline for features → MSM → FES.
Outputs saved in example_programs/programs_outputs/find_conformations_with_msm."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import mdtraj as md
import numpy as np

from pmarlo import Protein, api
from pmarlo.reporting.export import write_conformations_csv_json
from pmarlo.reporting.plots import save_fes_contour, save_transition_matrix_heatmap

BASE_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BASE_DIR / "tests" / "data"
PDB_PATH = TESTS_DIR / "3gd8-fixed.pdb"
DCD_PATH = TESTS_DIR / "traj.dcd"

OUT_DIR = (
    Path(__file__).resolve().parent / "programs_outputs" / "find_conformations_with_msm"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_conformation_finder(
    feature_specs: List[str] | None = None,
    requested_pair: Optional[Tuple[str, str]] = None,
    lag: int = 10,
) -> None:
    Protein(str(PDB_PATH), ph=7.0, auto_prepare=False)
    traj = md.load(str(DCD_PATH), top=str(PDB_PATH))
    specs = feature_specs or ["phi_psi"]
    X, cols, periodic = api.compute_features(traj, feature_specs=specs)
    Y = api.reduce_features(X, method="vamp", lag=lag, n_components=3)
    labels = api.cluster_microstates(Y, method="minibatchkmeans", n_states=20)
    T, pi = api.build_msm_from_labels(
        [labels], n_states=int(np.max(labels) + 1), lag=lag
    )
    macrostates = api.compute_macrostates(T, n_macrostates=4)

    items: List[dict] = []
    if macrostates is not None:
        macro = macrostates[labels]
        pi_macro = api.macrostate_populations(pi, macrostates)
        T_macro = api.macro_transition_matrix(T, pi, macrostates)
        mfpt = api.macro_mfpt(T_macro)
        for m in range(len(pi_macro)):
            idxs = np.where(macro == m)[0]
            if idxs.size == 0:
                continue
            centroid = Y[idxs].mean(axis=0)
            rep = int(idxs[np.argmin(np.linalg.norm(Y[idxs] - centroid, axis=1))])
            path = OUT_DIR / f"macro_{m:02d}.pdb"
            traj[rep].save_pdb(str(path))
            items.append(
                {
                    "type": "MSM",
                    "macrostate": int(m),
                    "representative_frame": rep,
                    "population": float(pi_macro[m]),
                    "mfpt_to": {
                        str(j): float(mfpt[m, j]) for j in range(mfpt.shape[1])
                    },
                    "rep_pdb": str(path),
                }
            )

    fes_info = api.generate_fes_and_pick_minima(
        X, cols, periodic, requested_pair=requested_pair
    )
    F = fes_info["fes"]
    save_fes_contour(
        F.F,
        F.xedges,
        F.yedges,
        fes_info["names"][0],
        fes_info["names"][1],
        str(OUT_DIR),
        f"fes_{fes_info['names'][0]}_vs_{fes_info['names'][1]}.png",
    )

    save_transition_matrix_heatmap(T, str(OUT_DIR), name="transition_matrix.png")
    write_conformations_csv_json(str(OUT_DIR), items)


if __name__ == "__main__":
    run_conformation_finder()
