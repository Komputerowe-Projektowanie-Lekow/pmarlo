# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
All Capabilities Demo: REMD → MSM → FES → Conformations and Plots

This comprehensive example showcases current package capabilities:
- Replica Exchange Molecular Dynamics (REMD) with demultiplexing fallback
- Markov State Model (MSM) analysis, ITS, plots, state table and representatives
- 2D Free Energy Surface (FES) generation and plotting
- Conformation finding from MSM macrostates and FES minima (CSV/JSON)

Outputs are written to:
  example_programs/programs_outputs/all_capabilities

Notes:
- Uses bundled test assets under `tests/data/` by default.
- Avoids requiring PDBFixer by using `3gd8-fixed.pdb`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np

from pmarlo import (
    MarkovStateModel,
    Protein,
    ReplicaExchange,
    api,
    power_of_two_temperature_ladder,
)
from pmarlo.reduce.reducers import vamp_reduce
from pmarlo.replica_exchange.config import RemdConfig
from pmarlo.reporting.export import write_conformations_csv_json
from pmarlo.reporting.plots import save_fes_contour, save_transition_matrix_heatmap

# ------------------------------ Configuration ------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BASE_DIR / "tests" / "data"
DEFAULT_PDB = TESTS_DIR / "3gd8-fixed.pdb"

OUT_DIR = Path(__file__).resolve().parent / "programs_outputs" / "all_capabilities"


# ------------------------------ Utilities ------------------------------


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize(name: str) -> str:
    return name.replace(":", "-").replace(" ", "_")


# ------------------------------ REMD ------------------------------


def run_remd(
    pdb_file: Path,
    output_dir: Path,
    temperatures: List[float],
    total_steps: int,
) -> Tuple[List[str], List[float]]:
    logging.info("Running REMD with temperatures: %s", temperatures)
    remd_out = output_dir / "replica_exchange"

    # Choose stride to target ~5000 frames per replica (best effort)
    equil = min(total_steps // 10, 200 if total_steps <= 2000 else 2000)
    dcd_stride = max(1, int(total_steps // 5000))
    exchange_frequency = max(100, total_steps // 20)

    remd = ReplicaExchange.from_config(
        RemdConfig(
            pdb_file=str(pdb_file),
            temperatures=temperatures,
            output_dir=str(remd_out),
            exchange_frequency=exchange_frequency,
            auto_setup=False,
            dcd_stride=dcd_stride,
        )
    )
    remd.plan_reporter_stride(
        total_steps=int(total_steps), equilibration_steps=int(equil), target_frames=5000
    )
    remd.setup_replicas()
    remd.run_simulation(total_steps=int(total_steps), equilibration_steps=int(equil))

    # Try to demultiplex to ~300 K; fall back to per-replica DCDs
    demuxed = remd.demux_trajectories(
        target_temperature=300.0, equilibration_steps=int(equil)
    )
    if demuxed:
        try:
            traj = md.load(str(demuxed), top=str(pdb_file))
            reporter_stride = getattr(remd, "reporter_stride", None)
            eff_stride = int(
                reporter_stride
                if reporter_stride
                else max(1, getattr(remd, "dcd_stride", 1))
            )
            production_steps = max(0, int(total_steps) - int(equil))
            expected = max(1, production_steps // eff_stride)
            if traj.n_frames >= expected:
                logging.info("Using demultiplexed trajectory: %s", demuxed)
                return [str(demuxed)], [300.0]
            logging.info(
                "Demux yielded %d frames (<%d); using multi-replica analysis",
                traj.n_frames,
                expected,
            )
        except Exception:
            logging.info(
                "Could not load demuxed trajectory reliably; using multi-replica analysis"
            )

    traj_files = [str(f) for f in remd.trajectory_files]
    logging.info("Using %d replica trajectories", len(traj_files))
    return traj_files, temperatures


# ------------------------------ MSM + FES ------------------------------


def run_msm_full(
    trajectory_files: List[str],
    topology_pdb: Path,
    output_dir: Path,
    feature_type: str,
    analysis_temperatures: Optional[List[float]],
    use_effective_for_uncertainty: bool = True,
    use_tica: bool = True,
) -> Path:
    logging.info("Building MSM and generating plots ...")
    msm_out = output_dir / "msm_analysis"

    msm = MarkovStateModel(
        trajectory_files=trajectory_files,
        topology_file=str(topology_pdb),
        temperatures=analysis_temperatures or [300.0],
        output_dir=str(msm_out),
    )
    # Set counting mode: sliding for ML; switch to effective when doing uncertainty
    if use_effective_for_uncertainty:
        msm.count_mode = "sliding"  # ML for main fit
    msm.load_trajectories()
    ft = feature_type
    if use_tica and ("tica" not in feature_type.lower()):
        ft = f"{feature_type}_tica"
    msm.compute_features(feature_type=ft)

    # Adapt clusters to data volume
    total_frames = 0
    try:
        total_frames = sum(t.n_frames for t in msm.trajectories)
    except Exception:
        total_frames = 0
    adaptive_states = max(5, min(50, total_frames // 50)) if total_frames > 0 else 50
    msm.cluster_features(n_clusters=int(adaptive_states))

    method = (
        "tram"
        if analysis_temperatures
        and len(analysis_temperatures) > 1
        and len(trajectory_files) > 1
        else "standard"
    )
    # First pass to compute ITS and choose lag
    candidate_lags = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]
    msm.build_msm(lag_time=5, method=method)
    msm.compute_implied_timescales(lag_times=candidate_lags, n_timescales=3)

    chosen_lag = 10
    try:
        lags = np.array(msm.implied_timescales["lag_times"])  # type: ignore[index]
        its = np.array(msm.implied_timescales["timescales"])  # type: ignore[index]
        scores: List[float] = []
        for idx in range(len(lags)):
            if idx == 0:
                scores.append(float("inf"))
                continue
            prev = its[idx - 1]
            cur = its[idx]
            mask = np.isfinite(prev) & np.isfinite(cur) & (np.abs(prev) > 0)
            if np.count_nonzero(mask) == 0:
                scores.append(float("inf"))
                continue
            rel = float(np.mean(np.abs((cur[mask] - prev[mask]) / prev[mask])))
            scores.append(rel)
        start_idx = min(3, len(scores) - 1)
        region = scores[start_idx:]
        if region:
            min_idx = int(np.nanargmin(region)) + start_idx
            chosen_lag = int(lags[min_idx])
    except Exception:
        chosen_lag = 10

    msm.build_msm(lag_time=chosen_lag, method=method)

    # Deeptime-backed diagnostics and analyses
    # 1) CK tests (micro and macro) saved as JSON
    try:
        macro_ck = msm.compute_ck_test_macrostates(n_macrostates=3, factors=[2, 3, 4])
    except Exception:
        macro_ck = None
    try:
        micro_ck = msm.compute_ck_test_micro(factors=[2, 3, 4])
    except Exception:
        micro_ck = None
    try:
        import json  # type: ignore

        with open(msm_out / "ck_tests.json", "w", encoding="utf-8") as f:
            json.dump({"macro": macro_ck, "micro": micro_ck}, f, indent=2)
    except Exception:
        pass

    # 1b) CK plot using deeptime utilities when available
    try:
        from deeptime.markov import TransitionCountEstimator  # type: ignore
        from deeptime.markov.msm import MaximumLikelihoodMSM  # type: ignore
        from deeptime.plots import plot_ck_test  # type: ignore
        from deeptime.util.validation import ck_test  # type: ignore

        # Build models across a small grid for visualization
        _lags = [max(1, chosen_lag), max(1, chosen_lag * 2), max(1, chosen_lag * 3)]
        models = []
        for L in _lags:
            C = (
                TransitionCountEstimator(lagtime=int(L), count_mode="sliding")
                .fit(msm.dtrajs)
                .fetch_model()
            )
            models.append(MaximumLikelihoodMSM(reversible=True).fit(C).fetch_model())
        ckobj = ck_test(models=models, n_metastable_sets=3)
        try:
            import matplotlib.pyplot as _plt  # type: ignore

            fig = plot_ck_test(ckobj)
            fig.savefig(msm_out / "ck_plot.png", dpi=200)
            _plt.close(fig)
        except Exception:
            pass
    except Exception:
        pass

    # 2) Bayesian MSM uncertainty (ITS/populations)
    try:
        # Flip to effective counts for uncertainty sampling
        bayes = msm.sample_bayesian_timescales(
            n_samples=200,
            count_mode=("effective" if use_effective_for_uncertainty else "sliding"),
        )
        if bayes is not None:
            import numpy as _np  # type: ignore

            out: Dict[str, Any] = {}
            if "timescales_samples" in bayes:
                ts = _np.asarray(bayes["timescales_samples"], dtype=float)
                q = _np.nanpercentile(ts, [2.5, 50.0, 97.5], axis=0)
                out["timescales_ci"] = {
                    "low": q[0].tolist(),
                    "median": q[1].tolist(),
                    "high": q[2].tolist(),
                }
            if "population_samples" in bayes:
                ps = _np.asarray(bayes["population_samples"], dtype=float)
                q = _np.nanpercentile(ps, [2.5, 50.0, 97.5], axis=0)
                out["populations_ci"] = {
                    "low": q[0].tolist(),
                    "median": q[1].tolist(),
                    "high": q[2].tolist(),
                }
            if out:
                with open(
                    msm_out / "bayesian_uncertainty.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(out, f, indent=2)
    except Exception:
        pass

    # 3) TPT/MFPT/committor between two most populated macrostates (if available)
    try:
        table = msm.create_state_table()
        if "macrostate" in table.columns:
            # Determine two largest macrostates by population
            pop_by_macro = (
                table.groupby("macrostate")["population"]
                .sum()
                .sort_values(ascending=False)
            )
            macro_ids = [int(i) for i in list(pop_by_macro.index[:2])]
            # Build microstate sets A/B
            micro_labels = table["macrostate"].to_numpy().astype(int)
            A = [int(i) for i in _np.where(micro_labels == macro_ids[0])[0]]  # type: ignore[name-defined]
            B = [int(i) for i in _np.where(micro_labels == macro_ids[1])[0]]  # type: ignore[name-defined]
            mfpt_val = msm.mfpt(A, B)
            comm_f = msm.committor(A, B)
            tpt = msm.tpt_flux(A, B)
            with open(msm_out / "tpt_summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "A_macro": macro_ids[0],
                        "B_macro": macro_ids[1],
                        "mfpt": float(mfpt_val) if mfpt_val is not None else None,
                        "committor_forward_head": (
                            comm_f[:10].tolist() if isinstance(comm_f, _np.ndarray) else None  # type: ignore[name-defined]
                        ),
                        "tpt_keys": list(tpt.keys()) if isinstance(tpt, dict) else None,
                    },
                    f,
                    indent=2,
                )
            # Persist TPT details when available
            try:
                if isinstance(tpt, dict):
                    import numpy as _np  # type: ignore

                    if "flux" in tpt and tpt["flux"] is not None:
                        _np.save(msm_out / "tpt_flux.npy", _np.asarray(tpt["flux"]))
                    if (
                        "committor_forward" in tpt
                        and tpt["committor_forward"] is not None
                    ):
                        _np.save(
                            msm_out / "committor_forward.npy",
                            _np.asarray(tpt["committor_forward"]),
                        )
                    if (
                        "committor_backward" in tpt
                        and tpt["committor_backward"] is not None
                    ):
                        _np.save(
                            msm_out / "committor_backward.npy",
                            _np.asarray(tpt["committor_backward"]),
                        )
            except Exception:
                pass
    except Exception:
        pass

    # FES (phi, psi) with adaptive bins
    adaptive_bins = max(20, min(50, int((total_frames or 0) ** 0.5))) or 20
    msm.generate_free_energy_surface(
        cv1_name="phi", cv2_name="psi", bins=int(adaptive_bins), temperature=300.0
    )

    # Save plots and artifacts
    msm.plot_free_energy_surface(save_file="free_energy_surface", interactive=False)
    msm.plot_implied_timescales(save_file="implied_timescales")
    msm.plot_free_energy_profile(save_file="free_energy_profile")
    msm.create_state_table()
    msm.extract_representative_structures(save_pdb=True)
    msm.save_analysis_results()

    # 4) VAMP diagnostic scatter (first two components) if features are present
    try:
        if msm.features is not None and msm.features.shape[0] > 1:
            Yv = vamp_reduce(msm.features, lag=max(1, msm.lag_time), n_components=2)
            try:
                import matplotlib.pyplot as _plt  # type: ignore

                _plt.figure(figsize=(6, 5))
                _plt.scatter(Yv[:, 0], Yv[:, 1], s=3, alpha=0.3, c="tab:green")
                _plt.xlabel("VAMP-1")
                _plt.ylabel("VAMP-2")
                _plt.title("VAMP projection (2D)")
                _plt.tight_layout()
                _plt.savefig(msm_out / "vamp_scatter.png", dpi=200)
                _plt.close()
            except Exception:
                pass
    except Exception:
        pass

    logging.info("MSM analysis complete: %s", msm_out)
    return msm_out


# ------------------------------ Conformations (API) ------------------------------


def run_conformations_api(
    topology_pdb: Path,
    trajectory_choice: str,
    output_dir: Path,
    feature_specs: Optional[List[str]] = None,
    requested_pair: Optional[Tuple[str, str]] = None,
) -> None:
    logging.info("Running API-based conformation finder on %s", trajectory_choice)
    out = output_dir

    traj = md.load(str(trajectory_choice), top=str(topology_pdb))

    # 1) Features → reduction → clustering via new API
    specs = feature_specs if feature_specs is not None else ["phi_psi"]
    X, cols, periodic = api.compute_features(traj, feature_specs=specs)
    # Prefer VAMP for robust kinetic projection
    Y = api.reduce_features(X, method="vamp", lag=10, n_components=3)
    labels = api.cluster_microstates(Y, method="minibatchkmeans", n_clusters=20)

    # 2) MSM from labels → macrostates
    dtrajs = [labels]
    try:
        T, pi = api.build_msm_from_labels(
            dtrajs, n_states=int(np.max(labels) + 1), lag=10
        )
    except Exception:
        # Fallback if labeling degenerate
        T, pi = api.build_msm_from_labels(
            dtrajs, n_states=max(5, int(np.max(labels) + 1)), lag=10
        )
    macrostates = api.compute_macrostates(T, n_macrostates=4)
    _ = save_transition_matrix_heatmap(T, str(out), name="transition_matrix.png")

    items: List[dict] = []
    if macrostates is not None:
        macro_of_micro = macrostates
        macro_per_frame = macro_of_micro[labels]
        pi_macro = api.macrostate_populations(pi, macro_of_micro)
        T_macro = api.macro_transition_matrix(T, pi, macro_of_micro)
        mfpt = api.macro_mfpt(T_macro)

        # Representatives per macrostate
        for macro_id in sorted(set(int(m) for m in macro_per_frame)):
            idxs = np.where(macro_per_frame == macro_id)[0]
            if idxs.size == 0:
                continue
            centroid = np.mean(Y[idxs], axis=0)
            deltas = np.linalg.norm(Y[idxs] - centroid, axis=1)
            best_local = int(idxs[int(np.argmin(deltas))])
            best_local = int(best_local % max(1, traj.n_frames))
            rep_path = out / f"macrostate_{macro_id:02d}_rep.pdb"
            try:
                traj[best_local].save_pdb(str(rep_path))
            except Exception:
                pass
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

    # 3) FES on selected pair using helpers; always produce an FES with sensible fallback
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
    names = fes_info["names"]
    fes = fes_info["fes"]
    minima = fes_info["minima"]
    # Save FES figure
    fname = f"fes_{_sanitize(names[0])}_vs_{_sanitize(names[1])}.png"
    _ = save_fes_contour(
        fes["F"], fes["xedges"], fes["yedges"], names[0], names[1], str(out), fname
    )

    for idx, entry in enumerate(minima.get("minima", [])):
        frames = entry.get("frames", [])
        if not frames:
            continue
        best_local = int(frames[0])
        rep_path = out / f"state_{idx:02d}_rep.pdb"
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

    write_conformations_csv_json(str(out), items)


# ------------------------------ Main ------------------------------


if __name__ == "__main__":
    configure_logging(verbose=True)

    pdb_path = DEFAULT_PDB.resolve()
    out_dir = ensure_output_dir(OUT_DIR)

    # Reasonable demo defaults
    steps = 2000  # keep fast; adjust higher for more robust MSM/FES
    feature_type = "phi_psi"
    temperatures = power_of_two_temperature_ladder(300.0, 375.0, 16)

    logging.info("Initializing protein: %s", pdb_path)
    _protein = Protein(str(pdb_path), ph=7.0, auto_prepare=False)

    # 1) REMD
    traj_files, analysis_temps = run_remd(
        pdb_file=pdb_path,
        output_dir=out_dir,
        temperatures=temperatures,
        total_steps=steps,
    )

    # 2) MSM + plots
    msm_dir = run_msm_full(
        trajectory_files=traj_files,
        topology_pdb=pdb_path,
        output_dir=out_dir,
        feature_type=feature_type,
        analysis_temperatures=analysis_temps,
    )

    # 3) Conformations & FES using API on a single trajectory
    chosen_traj = traj_files[0]
    run_conformations_api(
        topology_pdb=pdb_path,
        trajectory_choice=chosen_traj,
        output_dir=out_dir,
        feature_specs=["phi_psi"],
        requested_pair=None,
    )

    print("\n=== All Capabilities Demo completed ===")
    print(f"MSM analysis directory: {msm_dir}")
    print(f"Output base directory:  {out_dir}")
    print(
        "Saved files include: free_energy_surface.png, implied_timescales.png, free_energy_profile.png,"
    )
    print(
        "                     transition_matrix.png, fes_*.png, state_*_rep.pdb, macrostate_*_rep.pdb,"
    )
    print("                     conformations_summary.csv and states.json")
