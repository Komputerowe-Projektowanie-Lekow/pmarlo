"""
Complete pipeline: Replica Exchange → MSM (micro+macro) → FES → Conformations.

This example exposes an importable function (no CLI/env parsing):
    run_conformation_finder(feature_specs=None, requested_pair=None, use_remd=False,
                            total_steps=2000)

Outputs: example_programs/programs_outputs/free_energy_conformations_msm
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import mdtraj as md
import numpy as np

from pmarlo import Protein, ReplicaExchange, api, power_of_two_temperature_ladder
from pmarlo.replica_exchange.config import RemdConfig
from pmarlo.reporting.export import write_conformations_csv_json
from pmarlo.reporting.plots import save_fes_contour, save_transition_matrix_heatmap

BASE_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BASE_DIR / "tests" / "data"
PDB_PATH = TESTS_DIR / "3gd8-fixed.pdb"
DCD_PATH = TESTS_DIR / "traj.dcd"

OUT_DIR = (
    Path(__file__).resolve().parent
    / "programs_outputs"
    / "free_energy_conformations_msm"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_short_remd(pdb_file: Path, total_steps: int = 1000) -> Path | None:
    if RemdConfig is None or ReplicaExchange is None:
        return None
    out = OUT_DIR / "replica_exchange"
    out.mkdir(parents=True, exist_ok=True)
    temps = power_of_two_temperature_ladder(300.0, 360.0, 8)
    remd = ReplicaExchange.from_config(
        RemdConfig(
            pdb_file=str(pdb_file),
            temperatures=temps,
            output_dir=str(out),
            auto_setup=False,
            exchange_frequency=100,
            dcd_stride=max(1, total_steps // 1000),
        )
    )
    remd.plan_reporter_stride(
        total_steps=total_steps,
        equilibration_steps=min(100, total_steps // 10),
        target_frames=1000,
    )
    remd.setup_replicas()
    remd.run_simulation(
        total_steps=total_steps, equilibration_steps=min(100, total_steps // 10)
    )
    # Try to demultiplex near 300 K
    demux = remd.demux_trajectories(
        target_temperature=300.0, equilibration_steps=min(100, total_steps // 10)
    )
    return Path(demux) if demux else None


def _sanitize(n: str) -> str:
    return n.replace(":", "-").replace(" ", "_")


def run_conformation_finder(
    feature_specs: List[str] | None = None,
    requested_pair: Optional[Tuple[str, str]] = None,
    use_remd: bool = False,
    total_steps: int = 2000,
) -> None:

    # Load topology
    Protein(str(PDB_PATH), ph=7.0, auto_prepare=False)  # for parity and quick check

    # Trajectory selection
    if use_remd:
        demux = run_short_remd(PDB_PATH, total_steps=int(total_steps))
        traj_path = demux if demux else DCD_PATH
    else:
        traj_path = DCD_PATH

    traj = md.load(str(traj_path), top=str(PDB_PATH))

    # 1) Features → reduction → clustering via new API
    specs = feature_specs if feature_specs is not None else ["phi_psi"]
    X, cols, periodic = api.compute_features(traj, feature_specs=specs)
    # Try VAMP for robustness; fallback remains in api if unavailable
    Y = api.reduce_features(X, method="vamp", lag=10, n_components=3)
    labels = api.cluster_microstates(Y, method="minibatchkmeans", n_clusters=20)

    # 2) MSM from labels → macrostates (PCCA+-like)
    dtrajs = [labels]  # single trajectory
    # Scan implied timescales to choose a lag on a plateau
    candidate_lags = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
    its_times: List[Tuple[int, List[float]]] = []
    for lag in candidate_lags:
        try:
            T_tmp, _ = api.build_msm_from_labels(
                dtrajs, n_states=int(np.max(labels) + 1), lag=int(lag)
            )
            evals = np.linalg.eigvals(T_tmp.T)
            evals = np.real(evals)
            evals_sorted = sorted(evals, key=lambda v: -abs(v))
            taus: List[float] = []
            for lam in evals_sorted[1:4]:
                if lam <= 0.0 or lam >= 1.0:
                    taus.append(float("nan"))
                else:
                    taus.append(float(-lag / np.log(lam)))
            its_times.append((int(lag), taus))
        except Exception:
            its_times.append((int(lag), [float("nan"), float("nan"), float("nan")]))
    chosen_lag = 10
    try:
        scores: List[float] = []
        for idx in range(len(its_times)):
            if idx == 0:
                scores.append(float("inf"))
                continue
            prev = np.array(its_times[idx - 1][1], dtype=float)
            cur = np.array(its_times[idx][1], dtype=float)
            mask = np.isfinite(prev) & np.isfinite(cur) & (np.abs(prev) > 0)
            if np.count_nonzero(mask) == 0:
                scores.append(float("inf"))
                continue
            rel = float(np.mean(np.abs((cur[mask] - prev[mask]) / prev[mask])))
            scores.append(rel)
        start_idx = min(2, len(scores) - 1)
        region = scores[start_idx:]
        if region:
            min_idx = int(np.nanargmin(region)) + start_idx
            chosen_lag = int(its_times[min_idx][0])
    except Exception:
        chosen_lag = 10
    # Build MSM at chosen lag and compute macrostates
    T, pi = api.build_msm_from_labels(
        dtrajs, n_states=int(np.max(labels) + 1), lag=int(chosen_lag)
    )
    macrostates = api.compute_macrostates(T, n_macrostates=4)
    # Save transition matrix heatmap and implied timescales plot
    save_transition_matrix_heatmap(
        T, str(OUT_DIR), name=f"transition_matrix_lag{chosen_lag}.png"
    )
    # Save CK results (macro) for quick diagnostic
    try:
        from pmarlo.markov_state_model.markov_state_model import EnhancedMSM

        # Build small MSM object from labels to reuse CK helpers
        msm_tmp = EnhancedMSM(trajectory_files=[], topology_file=str(PDB_PATH))
        msm_tmp.dtrajs = [labels]
        msm_tmp.n_states = int(np.max(labels) + 1)
        msm_tmp.lag_time = int(chosen_lag)
        ck = msm_tmp.compute_ck_test_macrostates(n_macrostates=3, factors=[2, 3])
        if ck.mse:
            import json as _json  # type: ignore

            with open(OUT_DIR / "ck_macro.json", "w", encoding="utf-8") as f:
                _json.dump(ck.to_dict(), f, indent=2)
    except Exception:
        pass
    try:
        import matplotlib.pyplot as plt  # type: ignore

        lags = [lag for lag, _ in its_times]
        its_arrays = np.array([vals for _, vals in its_times], dtype=float)
        plt.figure(figsize=(7, 4))
        for k in range(min(its_arrays.shape[1], 3)):
            plt.plot(lags, its_arrays[:, k], marker="o", label=f"ITS{k+1}")
        plt.xlabel("Lag time (frames)")
        plt.ylabel("Implied timescale")
        plt.title("Implied timescales vs lag")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "implied_timescales.png", dpi=200)
        plt.close()
    except Exception:
        pass

    # 3) Representative per macrostate (closest to macro centroid in Y)
    items: List[dict] = []
    if macrostates is not None:
        # Compute macro assignment per microstate label, then per-frame
        macro_of_micro = macrostates
        macro_per_frame = macro_of_micro[labels]
        # Aggregate populations and macro-level T, MFPT
        pi_macro = api.macrostate_populations(pi, macro_of_micro)
        # Sanity: populations must sum to 1 within tolerance
        if not np.isclose(np.sum(pi_macro), 1.0, atol=1e-6):
            raise RuntimeError(
                "Macrostate populations do not sum to 1.0; inconsistent lumping."
            )
        T_macro = api.macro_transition_matrix(T, pi, macro_of_micro)
        mfpt = api.macro_mfpt(T_macro)
        for macro_id in sorted(set(int(m) for m in macro_per_frame)):
            idxs = np.where(macro_per_frame == macro_id)[0]
            if idxs.size == 0:
                continue
            centroid = np.mean(Y[idxs], axis=0)
            deltas = np.linalg.norm(Y[idxs] - centroid, axis=1)
            best_local = int(idxs[int(np.argmin(deltas))])
            if best_local < 0 or best_local >= traj.n_frames:
                best_local = int(best_local % max(1, traj.n_frames))
            rep_path = OUT_DIR / f"macrostate_{macro_id:02d}_rep.pdb"
            traj[best_local].save_pdb(str(rep_path))
            # Compute mean CVs for frames in this macro (over the feature matrix X)
            cv_means = None
            try:
                cv_means = np.mean(X[idxs, :], axis=0).tolist()
            except Exception:
                cv_means = None
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
                    "cv_means": cv_means,
                    "rep_pdb": str(rep_path),
                }
            )

    # 4) FES on selected pair using api helpers; always produce an FES with sensible fallback
    # Adaptive binning based on frames (avoid overly sparse histograms)
    adaptive_bins = max(30, min(80, int((getattr(traj, "n_frames", 0) or 1) ** 0.5)))
    fes_info = api.generate_fes_and_pick_minima(
        X,
        cols,
        periodic,
        requested_pair=requested_pair,
        bins=(adaptive_bins, adaptive_bins),
        temperature=300.0,
        smoothing="cosine",
        deltaF_kJmol=3.0,
    )
    i = int(fes_info["i"])  # noqa: F841 - may be useful for downstream
    j = int(fes_info["j"])  # noqa: F841
    names = fes_info["names"]
    per_i, per_j = fes_info["periodic_flags"]
    fes = fes_info["fes"]
    minima = fes_info["minima"]
    items.append(
        {
            "type": "FES",
            "bins_x": int(len(fes["xedges"]) - 1),
            "bins_y": int(len(fes["yedges"]) - 1),
            "pair": {"x": names[0], "y": names[1]},
            "periodic": {"x": bool(per_i), "y": bool(per_j)},
        }
    )
    for idx, entry in enumerate(minima.get("minima", [])):
        frames = entry.get("frames", [])
        if not frames:
            continue
        best_local = int(frames[0])
        rep_path = OUT_DIR / f"state_{idx:02d}_rep.pdb"
        try:
            traj[best_local].save_pdb(str(rep_path))
        except Exception:
            pass
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
    # Save a FES plot with a filename reflecting the selected pair
    fname = f"fes_{_sanitize(names[0])}_vs_{_sanitize(names[1])}.png"
    _ = save_fes_contour(
        fes["F"], fes["xedges"], fes["yedges"], names[0], names[1], str(OUT_DIR), fname
    )

    write_conformations_csv_json(str(OUT_DIR), items)
    print("\n=== REMD + MSM + FES conformation demo ===")
    print(f"Mode: {'REMD' if use_remd else 'Fast demo'}")
    print(f"Topology: {PDB_PATH}")
    print(f"Trajectory: {traj_path}")
    print(f"Output:    {OUT_DIR}")
    print(
        "Saved: transition_matrix_lag*.png, implied_timescales.png, FES plot, representatives and summary CSV/JSON"
    )


if __name__ == "__main__":
    # Showcase default usage without CLI/env parsing
    run_conformation_finder()
