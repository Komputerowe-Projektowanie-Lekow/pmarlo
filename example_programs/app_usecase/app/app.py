from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Support running via `streamlit run app.py` (no package context)
try:  # first try relative imports when executed as a package module
    from .backend import (
        SimResult,
        aggregate_and_build_bundle,
        emit_from_trajs,
        emit_from_trajs_simple,
        extractor_factory,
        recompute_msm_from_shards,
        run_short_sim,
        choose_sim_seed,
    )
    from .plots import plot_fes, plot_msm
    from .state import read_manifest, write_manifest
    from .cv_hooks import default_deeptica_params
except Exception:  # pragma: no cover - UI convenience fallback
    import sys as _sys
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in _sys.path:
        _sys.path.insert(0, str(_HERE))
    from backend import (  # type: ignore
        SimResult,
        aggregate_and_build_bundle,
        emit_from_trajs,
        emit_from_trajs_simple,
        extractor_factory,
        recompute_msm_from_shards,
        run_short_sim,
        choose_sim_seed,
    )
    from plots import plot_fes, plot_msm  # type: ignore
    from state import read_manifest, write_manifest  # type: ignore
    from cv_hooks import default_deeptica_params  # type: ignore
from pmarlo.engine.build import BuildResult
from pmarlo.utils.cleanup import prune_workspace


# Global single-worker executor to serialize runs
EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _workspace_root() -> Path:
    """App-local workspace under app_usecase/app_output.

    Keeps all app artifacts (shards, bundles, models, logs) scoped to the app.
    """
    return (_app_root() / "app_output").resolve()


def _app_root() -> Path:
    # example_programs/app_usecase
    return Path(__file__).resolve().parent.parent


def _inputs_root() -> Path:
    """App-local inputs directory under app_usecase/app_intputs (spelling as requested)."""
    return (_app_root() / "app_intputs").resolve()


def _default_pdb() -> Path:
    # Prefer input folder; fallback to legacy location next to app_usecase
    p_inputs = _inputs_root() / "3gd8-fixed.pdb"
    if p_inputs.exists():
        return p_inputs.resolve()
    legacy = _app_root() / "3gd8-fixed.pdb"
    return legacy.resolve()


def _resolve_input_path(raw: str) -> Path:
    """Resolve user-provided path robustly relative to app location.

    Tries as-is, then relative to the app root, then by filename in app root.
    """
    p = Path(raw)
    if p.exists():
        return p.resolve()
    # try relative to inputs, then app root
    cand_inputs = _inputs_root() / p
    if cand_inputs.exists():
        return cand_inputs.resolve()
    # try relative to app root
    cand1 = _app_root() / p
    if cand1.exists():
        return cand1.resolve()
    cand2 = _app_root() / p.name
    if cand2.exists():
        return cand2.resolve()
    return p


def _scan_shards(ws: Path) -> List[Path]:
    d = ws / "shards"
    if not d.exists():
        return []
    # Recurse into run_* subfolders to collect all shard JSONs
    return sorted(d.rglob("*.json"))


def _latest_build_log(ws: Path) -> Path | None:
    logs = ws / "logs"
    if not logs.exists():
        return None
    files = sorted(logs.glob("build-*.log"))
    return files[-1] if files else None


def _latest_sim_log(ws: Path) -> Path | None:
    logs = ws / "logs"
    if not logs.exists():
        return None
    files = sorted(logs.glob("sim-*.log"))
    return files[-1] if files else None


def _update_manifest_after_build(ws: Path, res: BuildResult, bundle: Path, shard_ids: List[str], params: Dict[str, Any]) -> None:
    m = read_manifest(ws)
    m["params"] = dict(params)
    m["shards"] = {"count": int(len(shard_ids)), "last_ids": shard_ids[-20:]}
    m["last_build"] = {
        "bundle": str(bundle),
        "dataset_hash": res.metadata.dataset_hash,
        "digest": res.metadata.digest,
        "flags": res.flags,
        "time": res.metadata.to_dict().get("build_opts", {}).get("seed", None),
    }
    write_manifest(ws, m)


def _update_manifest_after_sim(ws: Path, shard_ids: List[str], params: Dict[str, Any]) -> None:
    m = read_manifest(ws)
    m["params"] = dict(params)
    # Count all shards present recursively for a global view
    all_shards = _scan_shards(ws)
    m["shards"] = {
        "count": int(len(all_shards)),
        "last_ids": [p.stem for p in all_shards][-20:],
    }
    # do not touch last_build here
    write_manifest(ws, m)


def main() -> None:
    st.set_page_config(page_title="PMARLO Sharded MSM App", layout="wide")
    st.title("PMARLO Sharded MSM App")
    # Light auto-refresh to pick up job completion without manual interaction
    try:
        st.autorefresh(interval=2000, key="auto_refresh")  # type: ignore[attr-defined]
    except Exception:
        pass

    # Success banners for background jobs
    try:
        msg_sim = st.session_state.get("last_sim_message")
        if msg_sim:
            st.success(str(msg_sim))
        msg_build = st.session_state.get("last_build_message")
        if msg_build:
            st.success(str(msg_build))
    except Exception:
        pass

    ws = _workspace_root()
    (ws / "shards").mkdir(parents=True, exist_ok=True)
    (ws / "bundles").mkdir(parents=True, exist_ok=True)
    (ws / "models").mkdir(parents=True, exist_ok=True)
    (ws / "logs").mkdir(parents=True, exist_ok=True)
    _inputs_root().mkdir(parents=True, exist_ok=True)

    # Sidebar-ish left column for controls
    # 2-column layout: controls on the left (narrower), plots on the right
    left, right = st.columns([1, 2])

    with left:
        st.header("Controls")
        pdb_path = st.text_input("PDB file", value=str(_default_pdb()))
        uploaded_pdb = st.file_uploader("or upload PDB", type=["pdb"], accept_multiple_files=False)
        selected_pdb = pdb_path
        if uploaded_pdb is not None:
            up_dir = _inputs_root() / "uploads"
            up_dir.mkdir(parents=True, exist_ok=True)
            dest = up_dir / uploaded_pdb.name
            with open(dest, "wb") as fh:
                fh.write(uploaded_pdb.getbuffer())
            selected_pdb = str(dest)
        ref_dcd = st.text_input("Reference DCD (optional)", value="")

        # Sim parameters
        with st.expander("Set simulation and build parameters", expanded=True):
            minT = st.number_input("Min T (K)", value=290.0, step=1.0)
            maxT = st.number_input("Max T (K)", value=350.0, step=1.0)
            nrep = st.number_input("Num replicas", value=4, step=1)
            steps = st.number_input("Total MD steps", value=5000, step=100)
            exch = st.number_input(
                "Exchange frequency (steps)", value=250, step=10,
                help="Recommended 250–1000 for longer runs"
            )
            lag = st.number_input("Lag (frames)", value=5, step=1)
            bins_cv1 = st.number_input("Bins for cv1 (Rg)", value=24, step=1)
            bins_cv2 = st.number_input("Bins for cv2 (RMSD)", value=24, step=1)
            seed_mode = st.radio(
                "Simulation seed",
                options=["fixed", "auto", "none"],
                index=2,
                help="fixed: use the provided seed; auto: new 32-bit seed per run; none: engine default",
            )
            fixed_seed_val = None
            if seed_mode == "fixed":
                fixed_seed_val = st.number_input("Fixed seed", value=123, step=1)
            build_seed = st.number_input(
                "Build/emit seed", value=123, step=1, help="Seed for emission and build reproducibility"
            )
            emit_stride = st.number_input("Emit stride (downsample)", value=1, step=1, min_value=1)
            # Temperature schedule selector
            schedule_mode = st.radio(
                "Temperature schedule",
                options=["auto-linear", "auto-geometric", "custom"],
                index=0,
                help="Choose how to construct the replica temperature ladder",
            )
            apply_ladder = st.checkbox("Apply ladder to run", value=True)
            temps_list: list[float] | None = None
            preview_msg = ""
            if schedule_mode == "auto-geometric":
                try:
                    from pmarlo.utils.replica_utils import geometric_ladder as _geom
                    arr = _geom(float(minT), float(maxT), int(max(2, nrep)))
                    temps_list = [float(x) for x in list(arr)]
                    preview_msg = ", ".join(f"{t:.1f}" for t in temps_list)
                except Exception as e:
                    st.warning(f"Geometric ladder error: {e}")
            elif schedule_mode == "custom":
                raw = st.text_area("Custom temperatures (comma/space-separated)", value="")
                if raw.strip():
                    try:
                        import re as _re
                        vals = [float(x) for x in _re.split(r"[\s,]+", raw.strip()) if x]
                        if len(vals) < 2 or any(v <= 0 for v in vals):
                            raise ValueError("Provide at least two positive values")
                        if any(vals[i] >= vals[i+1] for i in range(len(vals)-1)):
                            raise ValueError("Values must be strictly increasing")
                        temps_list = vals
                        preview_msg = f"{len(vals)} temps: {vals[0]:.1f}..{vals[-1]:.1f}"
                    except Exception as e:
                        st.error(f"Invalid custom list: {e}")
            else:
                # auto-linear preview
                import numpy as _np
                arr = _np.linspace(float(minT), float(maxT), int(max(2, nrep)))
                temps_list = [float(x) for x in arr]
                preview_msg = ", ".join(f"{t:.1f}" for t in temps_list)
            st.caption("Resolved ladder: " + preview_msg)

            # Starting structure controls
            sims_root = _workspace_root() / "sims"
            prev_runs = [""]
            if sims_root.exists():
                prev_runs += [str(p) for p in sorted(sims_root.glob("run-*"))]
            start_choice = st.radio(
                "Starting structure",
                options=["Initial PDB", "Last frame of run", "Random high‑T frame of run"],
                index=0,
            )
            start_run = ""
            if start_choice != "Initial PDB":
                start_run = st.selectbox("Run directory", options=prev_runs, index=0)
            # Ladder suggestion
            if st.button("Suggest ladder"):
                try:
                    from pmarlo.utils.replica_utils import geometric_temperature_ladder as _geom_ladder

                    ladder = _geom_ladder(float(minT), float(maxT), int(max(2, nrep)))
                    st.info(
                        "Geometric ladder: "
                        + ", ".join(f"{t:.1f}" for t in ladder)
                    )
                except Exception as e:
                    st.warning(f"Could not suggest ladder: {e}")
            jitter_start = st.checkbox("Jitter start positions", value=False, help="Apply small Gaussian noise to restart positions")
            jitter_sigma = st.number_input("Jitter sigma (Å)", value=0.05, step=0.01, min_value=0.0, format="%.2f")
            velocity_reseed = st.checkbox("Velocity reseed", value=True, help="Randomize initial velocities even when reusing coordinates")

            learn_cv = st.checkbox("Learn CVs (Deep-TICA)", value=False)
            deeptica_params: Dict[str, Any] = {}
            if learn_cv:
                deeptica_params["lag"] = st.number_input("DeepTICA lag", value=int(max(1, lag)))
                deeptica_params["n_out"] = st.number_input("DeepTICA outputs", value=2, step=1)
                deeptica_params["hidden"] = (64, 64)
                deeptica_params["max_epochs"] = 200
                deeptica_params["early_stopping"] = 20
                deeptica_params["reweight_mode"] = "scaled_time"

        # Run/build controls
        sim_fut_ref = st.session_state.get("sim_future")
        build_fut_ref = st.session_state.get("build_future")
        sim_running = bool(sim_fut_ref is not None and not getattr(sim_fut_ref, "done", lambda: True)())
        build_running = bool(build_fut_ref is not None and not getattr(build_fut_ref, "done", lambda: True)())
        any_running = bool(sim_running or build_running)

        if st.button("Start new simulation (emit shards)", disabled=any_running):
            if not any_running:
                # Clear previous messages
                st.session_state["last_sim_message"] = None
                st.session_state["sim_future"] = EXECUTOR.submit(
                    _run_sim_and_emit_job,
                    _resolve_input_path(selected_pdb),
                    _resolve_input_path(ref_dcd) if ref_dcd.strip() else None,
                    ws,
                    float(minT),
                    float(maxT),
                    int(nrep),
                    int(steps),
                    int(exch),
                    {"Rg": int(bins_cv1), "RMSD_ref": int(bins_cv2)},
                    str(seed_mode),
                    int(fixed_seed_val) if (seed_mode == "fixed" and fixed_seed_val is not None) else None,
                    int(build_seed),
                    int(emit_stride),
                    schedule_mode,
                    temps_list if apply_ladder else None,
                    (
                        "none"
                        if start_choice == "Initial PDB"
                        else ("last_frame" if start_choice.startswith("Last frame") else "random_highT")
                    ),
                    _resolve_input_path(start_run) if start_run else None,
                    bool(jitter_start),
                    float(jitter_sigma),
                    bool(velocity_reseed),
                )
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

        if st.button("Build MSM/FES from existing shards", disabled=any_running):
            if not any_running:
                st.session_state["last_build_message"] = None
                st.session_state["build_future"] = EXECUTOR.submit(
                    _build_from_existing,
                    ws,
                    int(lag),
                    {"Rg": int(bins_cv1), "RMSD_ref": int(bins_cv2)},
                    int(build_seed),
                    float((minT + maxT) / 2.0),
                    bool(learn_cv),
                    dict(deeptica_params or {}),
                )
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

        if st.button("Build MSM/FES with Deep‑TICA", disabled=any_running):
            if not any_running:
                dparams = default_deeptica_params(lag=int(max(1, lag)))
                st.session_state["last_build_message"] = None
                st.session_state["build_future"] = EXECUTOR.submit(
                    _build_from_existing,
                    ws,
                    int(lag),
                    {"Rg": int(bins_cv1), "RMSD_ref": int(bins_cv2)},
                    int(build_seed),
                    float((minT + maxT) / 2.0),
                    True,
                    dparams,
                )
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
        st.caption("Tip: Learn CVs is a build‑time option; no new simulation needed.")

        # Workspace cleanup controls
        st.subheader("Disk Cleanup")
        st.caption(
            "Prune large intermediates (DCDs, checkpoints, logs) after shards/bundles exist."
        )
        mode = st.selectbox("Prune mode", ["conservative", "aggressive"], index=0)
        dry_run = st.checkbox("Dry run (list only)", value=True)
        if st.button("Prune workspace", disabled=any_running):
            try:
                rep = prune_workspace(ws, mode=mode, dry_run=bool(dry_run))
                st.success(
                    f"Cleanup completed: removed={len(rep.removed)} kept={len(rep.kept)} errors={len(rep.errors)}"
                )
                with st.expander("Details", expanded=False):
                    st.json(rep.to_dict())
            except Exception as e:
                st.error(f"Cleanup failed: {e}")

        if st.button("Clear all data (outputs)", disabled=any_running):
            _clear_workspace(ws)
            try:
                st.session_state["last_sim_message"] = None
                st.session_state["last_build_message"] = None
            except Exception:
                pass
            try:
                st.experimental_rerun()
            except Exception:
                pass

        st.write("Sim status:", "running…" if sim_running else "idle")
        st.write("Build status:", "running…" if build_running else "idle")

        # Current info panel
        st.subheader("Current Info")
        man = read_manifest(ws)
        last = man.get("last_build")
        st.write("Shards:", man.get("shards", {}))
        if last:
            st.write("Last build digest:", last.get("digest"))
            st.write("Dataset hash:", last.get("dataset_hash"))
            st.write("Flags:", last.get("flags"))
        with st.expander("Build logs", expanded=False):
            logp = _latest_build_log(ws)
            if logp and logp.exists():
                try:
                    text = logp.read_text(encoding="utf-8")
                    st.code(text[-4000:] if len(text) > 4000 else text)
                except Exception as e:
                    st.write("(no logs)")
        with st.expander("Emit logs", expanded=False):
            logs = (ws / "logs")
            if logs.exists():
                files = sorted(logs.glob("emit-*.log"))
                if files:
                    try:
                        text = files[-1].read_text(encoding="utf-8")
                        st.code(text[-4000:] if len(text) > 4000 else text)
                    except Exception:
                        st.write("(no logs)")
        with st.expander("Sim logs", expanded=False):
            logp = _latest_sim_log(ws)
            if logp and logp.exists():
                try:
                    text = logp.read_text(encoding="utf-8")
                    st.code(text[-4000:] if len(text) > 4000 else text)
                except Exception:
                    st.write("(no logs)")

    # Right panel: plots
    with right:
        # Stack plots vertically for better proportions on wide screens
        bundle_paths = sorted((ws / "bundles").glob("*.json"))
        latest_bundle = bundle_paths[-1] if bundle_paths else None
        res: BuildResult | None = None
        if latest_bundle and latest_bundle.exists():
            try:
                res = BuildResult.from_json(latest_bundle.read_text(encoding="utf-8"))
            except Exception:
                res = None

        # MSM
        T = res.transition_matrix if (res and res.transition_matrix is not None) else None
        pi = res.stationary_distribution if (res and res.stationary_distribution is not None) else None
        if T is None or pi is None or T.size == 0:
            # try recomputing from shards using manifest-stored edges if present
            man = read_manifest(ws)
            notes = (res.metadata.applied_opts.notes if res else {}) or {}
            edges_map = notes.get("cv_bin_edges", {})
            if edges_map:
                edge_arrays = {k: np.asarray(v, dtype=float) for k, v in edges_map.items()}
                shards = _scan_shards(ws)
                if shards:
                    try:
                        T, pi = recompute_msm_from_shards(shards, edges_by_name=edge_arrays, lag=int(man.get("params", {}).get("lag", 5)))
                    except Exception:
                        T, pi = None, None
        st.subheader("Transition Matrix / π")
        fig1 = plot_msm(T, pi)
        st.pyplot(fig1, use_container_width=True)

        # FES
        fes = res.fes if res else None
        st.subheader("Free Energy Surface")
        fig2 = plot_fes(fes)
        st.pyplot(fig2, use_container_width=True)

        if res is not None:
            st.write("Digest:", res.metadata.digest)
            st.write("Dataset hash:", res.metadata.dataset_hash)
            st.write("Flags:", res.flags)
            # Show MLCV status if present
            mlcv = (res.artifacts or {}).get("mlcv_deeptica") if hasattr(res, "artifacts") else None
            if isinstance(mlcv, dict):
                if mlcv.get("applied"):
                    st.success("Learned CVs applied (Deep‑TICA)")
                elif mlcv.get("skipped"):
                    st.info(f"Learned CVs skipped: {mlcv.get('reason','unknown')}")

        # Diagnostics panel for last simulation
        st.subheader("REMD Diagnostics")
        sims = sorted((ws / "sims").glob("run-*"))
        last_run = sims[-1] if sims else None
        if last_run and last_run.exists():
            diag = last_run / "replica_exchange" / "exchange_diagnostics.json"
            if diag.exists():
                try:
                    import json as _json
                    d = _json.loads(diag.read_text(encoding="utf-8"))
                    st.write("Acceptance:", f"{d.get('acceptance_mean', float('nan')):.3f}")
                    st.write("Diffusion (per 10k steps):", f"{d.get('mean_abs_disp_per_10k_steps', float('nan')):.3f}")
                    # Small table of per-pair acceptance with temperatures
                    temps = [float(x) for x in d.get("temperatures", [])]
                    rates = d.get("acceptance_per_pair", [])
                    if temps and rates and len(temps) >= 2:
                        rows = [
                            {
                                "pair": f"{i}-{i+1}",
                                "T_i": f"{temps[i]:.1f}",
                                "T_j": f"{temps[i+1]:.1f}",
                                "acc": f"{float(rates[i]):.3f}" if i < len(rates) else "NA",
                            }
                            for i in range(min(len(temps) - 1, len(rates)))
                        ]
                        st.table(rows)
                    spark = d.get("sparkline", [])
                    if spark:
                        st.caption("Per-sweep mean |Δstate| (sparkline)")
                        try:
                            st.line_chart(spark, height=100)
                        except Exception:
                            st.write(spark[:10], "...")
                except Exception as e:
                    st.info(f"No diagnostics available: {e}")

    # Poll background jobs
    sim_fut = st.session_state.get("sim_future")
    if sim_fut is not None and sim_fut.done():
        try:
            sim_fut.result()
            st.session_state["last_sim_message"] = "Simulation and emission finished successfully. You can now Build MSM/FES."
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            try:
                print(f"[pmarlo] workflow: simulation failed: {e}", flush=True)
            except Exception:
                pass
        finally:
            st.session_state["sim_future"] = None
            try:
                st.experimental_rerun()
            except Exception:
                pass

    b_fut = st.session_state.get("build_future")
    if b_fut is not None and b_fut.done():
        try:
            b_fut.result()
            st.session_state["last_build_message"] = "Build finished successfully. Plots updated."
        except Exception as e:
            st.error(f"Build failed: {e}")
            try:
                print(f"[pmarlo] workflow: build failed: {e}", flush=True)
            except Exception:
                pass
        finally:
            st.session_state["build_future"] = None
            try:
                st.experimental_rerun()
            except Exception:
                pass


def _run_sim_and_emit_job(
    pdb: Path,
    ref_dcd: Path | None,
    ws: Path,
    minT: float,
    maxT: float,
    nrep: int,
    steps: int,
    exchange_frequency: int,
    bins: Dict[str, int],
    seed_mode: str,
    seed_fixed: int | None,
    build_seed: int,
    emit_stride: int,
    start_mode: str,
    start_run: Path | None,
    jitter_start: bool,
    jitter_sigma: float,
    velocity_reseed: bool,
) -> None:
    try:
        # 1) run sim
        temps = list(np.linspace(float(minT), float(maxT), int(max(2, nrep))))
        run_seed = choose_sim_seed(str(seed_mode), fixed=(int(seed_fixed) if seed_fixed is not None else None))
        sim = run_short_sim(
            Path(pdb),
            ws,
            temps,
            int(steps),
            quick=True,
            random_seed=run_seed,
            start_mode=str(start_mode),
            start_run=start_run,
            jitter_start=bool(jitter_start),
            jitter_sigma_A=float(jitter_sigma),
            velocity_reseed=bool(velocity_reseed),
            exchange_frequency_steps=int(exchange_frequency),
            temperature_schedule_mode=str(schedule_mode),
        )

        # 2) emit shards under unique subdir per run to avoid overwriting (simple API)
        shards_dir = ws / "shards" / Path(sim.run_dir).name
        shard_jsons = emit_from_trajs_simple(
            sim.traj_files,
            shards_dir,
            pdb=Path(pdb),
            ref_dcd=Path(ref_dcd) if ref_dcd else None,
            temperature=float(np.median(temps)),
            seed_start=int(build_seed),
            stride=int(max(1, emit_stride)),
            provenance={"sim_seed": (int(run_seed) if run_seed is not None else None), "seed_mode": str(seed_mode)},
        )

        # 3) update manifest (no build here)
        params = {
            "temperatures": temps,
            "steps": int(steps),
            "bins": bins,
            "seed_mode": str(seed_mode),
            "sim_seed": (int(run_seed) if run_seed is not None else None),
            "build_seed": int(build_seed),
            "exchange_frequency_steps": int(exchange_frequency),
            "temperature_schedule": {"mode": str(schedule_mode), "applied": bool(temps_list is not None)},
            "start_mode": str(start_mode),
            "start_run": str(start_run) if start_run else None,
            "jitter_start": bool(jitter_start),
            "jitter_sigma_A": float(jitter_sigma),
            "velocity_reseed": bool(velocity_reseed),
        }
        # Copy ladder and provenance into shard directory for convenience
        try:
            import shutil as _shutil

            lad = Path(sim.run_dir) / "replica_exchange" / "temps.txt"
            prov = Path(sim.run_dir) / "replica_exchange" / "provenance.json"
            if lad.exists():
                _shutil.copyfile(str(lad), str(shards_dir / "temps.txt"))
            if prov.exists():
                _shutil.copyfile(str(prov), str(shards_dir / "provenance.json"))
        except Exception:
            pass
        _update_manifest_after_sim(ws, [Path(p).stem for p in shard_jsons], params)
        # Explicit console cue for users following logs
        try:
            print(
                "[pmarlo] workflow: simulation + emission finished successfully; next → Build MSM/FES",
                flush=True,
            )
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"[pmarlo] workflow: simulation job error: {e}", flush=True)
        except Exception:
            pass
        raise


def _build_from_existing(
    ws: Path,
    lag: int,
    bins: Dict[str, int],
    seed: int,
    temperature: float,
    learn_cv: bool,
    deeptica_params: Dict[str, Any],
) -> None:
    try:
        shards = _scan_shards(ws)
        if not shards:
            # Raise to surface an error in the main UI thread instead of silently warning in a worker
            raise RuntimeError("No shards found to rebuild. Run a simulation or place shard JSONs under workspace/shards.")
        out_bundle = ws / "bundles" / f"build-rebuild-{len(shards)}-{np.random.randint(0, 1e6):06d}.json"
        res, ds_hash, edges = aggregate_and_build_bundle(
            shards,
            out_bundle,
            bins=bins,
            lag=int(lag),
            seed=int(seed),
            temperature=float(temperature),
            learn_cv=bool(learn_cv),
            deeptica_params=dict(deeptica_params or {}),
            workspace=ws,
        )
        params = {
            "lag": int(lag),
            "bins": bins,
            "seed": int(seed),
            "learn_cv": bool(learn_cv),
            "deeptica_params": dict(deeptica_params or {}),
            "cv_bin_edges": {k: v.tolist() for k, v in edges.items()},
        }
        _update_manifest_after_build(ws, res, out_bundle, [p.stem for p in shards], params)
        # Explicit console cue for users following logs
        try:
            print(
                "[pmarlo] workflow: build finished successfully; plots updated",
                flush=True,
            )
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"[pmarlo] workflow: build job error: {e}", flush=True)
        except Exception:
            pass
        raise


def _clear_workspace(ws: Path) -> None:
    # Remove standard subfolders and state.json
    for sub in ("shards", "bundles", "models", "logs", "sims"):
        p = ws / sub
        if p.exists():
            import shutil

            shutil.rmtree(p, ignore_errors=True)
    state = ws / "state.json"
    try:
        if state.exists():
            state.unlink()
    except Exception:
        pass


if __name__ == "__main__":
    main()
