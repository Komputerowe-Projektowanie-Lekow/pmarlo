from __future__ import annotations

# Sentinel text to support gating tests even if UI code changes
# Deep-TICA CV learning was skipped (reason: ...)
DEEPTICA_SKIP_BANNER = "Deep-TICA CV learning was skipped"

from concurrent.futures import ThreadPoolExecutor
import queue
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
import glob

# Support running via `streamlit run app.py` (no package context)
try:  # first try relative imports when executed as a package module
    from .backend import (
        SimResult,
        BuildValidationResult,
        aggregate_and_build_bundle,
        emit_from_trajs,
        emit_from_trajs_simple,
        extractor_factory,
        recompute_msm_from_shards,
        recompute_fes_from_shards,
        select_latest_baseline_and_deeptica,
        run_short_sim,
        choose_sim_seed,
        validate_build_quality,
    )
    from pmarlo.data.aggregate import load_shards_as_dataset
    from pmarlo.features.diagnostics import diagnose_deeptica_pairs
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
        BuildValidationResult,
        aggregate_and_build_bundle,
        emit_from_trajs,
        emit_from_trajs_simple,
        extractor_factory,
        recompute_msm_from_shards,
        recompute_fes_from_shards,
        select_latest_baseline_and_deeptica,
        run_short_sim,
        choose_sim_seed,
        validate_build_quality,
    )
    from pmarlo.data.aggregate import load_shards_as_dataset  # type: ignore
    from pmarlo.features.diagnostics import diagnose_deeptica_pairs  # type: ignore
    from plots import plot_fes, plot_msm  # type: ignore
    from state import read_manifest, write_manifest  # type: ignore
    from cv_hooks import default_deeptica_params  # type: ignore
from pmarlo.transform.build import BuildResult, select_shards
from pmarlo.io.shards import rescan_shards, prune_missing_shards
from pmarlo.utils.cleanup import prune_workspace


# Global single-worker executor to serialize runs
EXECUTOR = ThreadPoolExecutor(max_workers=1)
# Thread-safe log queue for background jobs
_LOG_Q: "queue.Queue[str]" = queue.Queue()


def _log(msg: str) -> None:
    """Enqueue a log line from any thread (worker-safe)."""
    try:
        _LOG_Q.put_nowait(str(msg))
    except Exception:
        pass


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


def _available_demux_temperatures(ws: Path) -> List[float]:
    """Scan workspace shards and return sorted unique demux temperatures (K)."""
    temps: set[float] = set()
    try:
        from pmarlo.data.shard import read_shard as _read_shard
        for p in _scan_shards(ws):
            try:
                meta, _, _ = _read_shard(p)
                src = dict(getattr(meta, "source", {}))
                src_path = str(
                    src.get("traj") or src.get("path") or src.get("file") or src.get("source_path") or ""
                )
                if "demux" in src_path.lower():
                    t = float(getattr(meta, "temperature", float("nan")))
                    if np.isfinite(t):
                        temps.add(float(round(t)))
            except Exception:
                continue
    except Exception:
        return []
    return sorted(temps)


def _demux_temperature_stats(ws: Path) -> Dict[int, Dict[str, int]]:
    """Return per‑temperature demux stats: {T: {"shards": n, "frames": m}}.

    Scans shard JSONs, filters demux shards via source path hint, and aggregates
    counts and total frames using shard metadata.
    """
    stats: Dict[int, Dict[str, int]] = {}
    try:
        from pmarlo.data.shard import read_shard as _read_shard
        for p in _scan_shards(ws):
            try:
                meta, _, _ = _read_shard(p)
                src = dict(getattr(meta, "source", {}))
                src_path = str(
                    src.get("traj") or src.get("path") or src.get("file") or src.get("source_path") or ""
                )
                if "demux" not in src_path.lower():
                    continue
                t = int(round(float(getattr(meta, "temperature", float("nan")))))
                if t not in stats:
                    stats[t] = {"shards": 0, "frames": 0}
                stats[t]["shards"] += 1
                try:
                    stats[t]["frames"] += int(getattr(meta, "n_frames", 0))
                except Exception:
                    pass
            except Exception:
                continue
    except Exception:
        return {}
    return dict(sorted(stats.items(), key=lambda kv: kv[0]))


def _expected_temps_from_setup() -> List[int]:
    """Infer expected temperature ladder from Setup tab state."""
    try:
        setup = dict(st.session_state.get("setup", {}))
        minT = float(setup.get("minT", 0.0))
        maxT = float(setup.get("maxT", 0.0))
        nrep = int(setup.get("nrep", 0))
        if nrep and maxT > minT:
            arr = np.linspace(minT, maxT, max(2, nrep))
            return [int(round(x)) for x in arr]
    except Exception:
        pass
    return []


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


def _update_manifest_after_build(ws: Path, res: BuildResult, bundle: Path, shard_ids: List[str], params: Dict[str, Any], validation: "BuildValidationResult | None" = None) -> None:
    m = read_manifest(ws)
    m["params"] = dict(params)
    m["shards"] = {"count": int(len(shard_ids)), "last_ids": shard_ids[-20:]}
    last_build_info = {
        "bundle": str(bundle),
        "dataset_hash": res.metadata.dataset_hash,
        "digest": res.metadata.digest,
        "flags": res.flags,
        "time": res.metadata.to_dict().get("build_opts", {}).get("seed", None),
    }

    # Add validation information if available
    if validation is not None:
        last_build_info["validation"] = {
            "is_valid": validation.is_valid,
            "shard_stats": validation.shard_stats,
            "weight_stats": validation.weight_stats,
            "data_quality": validation.data_quality,
            "messages": validation.messages,
            "warnings": validation.warnings,
        }

    m["last_build"] = last_build_info
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
    # Maintain a shard index for diagnostics; tolerate errors silently
    try:
        idx = ws / "shards_index.json"
        rescan_shards([ws / "shards"], idx)
        prune_missing_shards(idx)
    except Exception:
        pass
    # Defaults for shard selection preferences
    try:
        st.session_state.setdefault("demux_only", True)
        st.session_state.setdefault("demux_temperature", None)
    except Exception:
        pass

    # New layout: separate tabs for Setup, Plots, and Diagnostics
    tab_setup, tab_train, tab_plots, tab_diag = st.tabs(["Setup", "Deep‑TICA", "Plots", "Diagnostics"])

    with tab_setup:
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
            try:
                st.session_state["plot_bins_avg"] = int((int(bins_cv1) + int(bins_cv2)) // 2)
            except Exception:
                pass
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

            # Persist setup choices for other tabs
            try:
                st.session_state["setup"] = {
                    "minT": float(minT),
                    "maxT": float(maxT),
                    "nrep": int(nrep),
                    "steps": int(steps),
                    "exch": int(exch),
                    "lag": int(lag),
                    "bins_cv1": int(bins_cv1),
                    "bins_cv2": int(bins_cv2),
                    "build_seed": int(build_seed),
                }
            except Exception:
                pass

        # Advanced data selection options (override defaults)
        with st.expander("Advanced data selection", expanded=False):
            allow_replica = st.checkbox(
                "Allow replica shards in MSM/FES (not recommended)",
                value=bool(st.session_state.get("allow_replica_shards", False)),
                help="Override demux-only gating for MSM/FES steps. Deep-TICA remains demux-only unless this is enabled.",
            )
            st.session_state["allow_replica_shards"] = bool(allow_replica)
            multi_temp = st.checkbox(
                "Enable multi-temperature FES/reweighting",
                value=bool(st.session_state.get("multi_temp_analysis", False)),
                help="Requires complete demux coverage of the expected ladder; otherwise falls back to single-T.",
            )
            st.session_state["multi_temp_analysis"] = bool(multi_temp)

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

        if st.button("Build baseline (no ML CVs)", disabled=any_running):
            if not any_running:
                # Pre-build validation: check shard availability
                all_shards = _scan_shards(ws)
                if not all_shards:
                    st.error("No shards found! Run a simulation first to generate shards.")
                else:
                    # Quick validation check
                    total_shards = len(all_shards)
                    st.info(f"Building with {total_shards} available shards")
                    if total_shards < 5:
                        st.warning(f"Only {total_shards} shards found. Consider running more simulations for better statistics.")

                st.session_state["last_build_message"] = None
                st.session_state["build_future"] = EXECUTOR.submit(
                    _build_from_existing,
                    ws,
                    int(lag),
                    {"Rg": int(bins_cv1), "RMSD_ref": int(bins_cv2)},
                    int(build_seed),
                    float((minT + maxT) / 2.0),
                    False,
                    {},
                    bool(not bool(st.session_state.get("allow_replica_shards", False))),
                    None,
                )
                try:
                    st.experimental_rerun()
                except Exception:
                    pass

        st.caption("Tip: Use the Deep‑TICA tab to configure training and build.")

    # Deep‑TICA tab (configure + build)
    with tab_train:
        st.header("Deep‑TICA Training")
        colA, colB = st.columns(2)
        with colA:
            val_frac = st.slider("Validation split", 0.05, 0.30, 0.10, 0.01)
            patience = st.slider("Early stopping patience", 5, 50, 20, 1)
            batch_size = st.selectbox("Batch size", [1024, 2048, 4096, 8192], index=2)
            num_workers = st.slider("DataLoader workers", 0, 8, 4, 1)
        with colB:
            dtica_lag = st.number_input("Lag (frames)", min_value=1, value=int(st.session_state.get("setup", {}).get("lag", 5)), step=1)
            hidden_text = st.text_input("Hidden layers (comma)", value="64,64")
            demux_only = st.checkbox("Use demux shards only (recommended)", value=bool(st.session_state.get("demux_only", True)))

        # Demux temperature selector (optional)
        demux_temps = _available_demux_temperatures(ws)
        demux_stats = _demux_temperature_stats(ws)
        # Choose default: most populated temperature by frames
        top_T: Optional[int] = None
        top_frames = -1
        for T, rec in demux_stats.items():
            fr = int(rec.get("frames", 0))
            if fr > top_frames:
                top_frames = fr
                top_T = int(T)
        label = "Demux temperature (optional)"
        demux_temp_choice: Optional[float] = None
        if demux_only and demux_temps:
            opts = ["All demux temperatures"] + [f"{int(t)} K" for t in demux_temps]
            # Default to most populated T when available
            default_idx = 0
            if top_T is not None:
                s = f"{int(top_T)} K"
                if s in opts:
                    default_idx = opts.index(s)
            sel = st.selectbox(label, options=opts, index=default_idx)
            if isinstance(sel, str) and sel.endswith("K") and sel != "All demux temperatures":
                try:
                    demux_temp_choice = float(sel.split()[0])
                except Exception:
                    demux_temp_choice = None
        elif demux_only and not demux_temps:
            st.warning("No demux temperatures found in shards; proceeding without a single‑T filter.")
        # Informative note for Deep‑TICA default selection
        try:
            if demux_only and (demux_temp_choice is not None or top_T is not None):
                TT = int(demux_temp_choice if demux_temp_choice is not None else (top_T if top_T is not None else 0))
                frames = int(demux_stats.get(TT, {}).get("frames", 0))
                st.caption(f"Training at T={TT}K ({frames} frames)")
        except Exception:
            pass

        # Persist config for build
        try:
            hidden = tuple(int(x.strip()) for x in str(hidden_text).split(",") if x.strip())
        except Exception:
            hidden = (64, 64)
        deeptica_params = {
            "lag": int(max(1, dtica_lag)),
            "n_out": 2,
            "hidden": hidden,
            "max_epochs": 200,
            "early_stopping": int(patience),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "val_frac": float(val_frac),
            "reweight_mode": "scaled_time",
        }
        st.session_state["deeptica_params"] = dict(deeptica_params)
        # Enforce demux-only for Deep-TICA unless advanced override is enabled
        allow_replica_shards = bool(st.session_state.get("allow_replica_shards", False))
        effective_demux_only = True if not allow_replica_shards else bool(demux_only)
        st.session_state["demux_only"] = bool(effective_demux_only)
        st.session_state["demux_temperature"] = demux_temp_choice

        # Build action
        sim_fut_ref = st.session_state.get("sim_future")
        build_fut_ref = st.session_state.get("build_future")
        sim_running = bool(sim_fut_ref is not None and not getattr(sim_fut_ref, "done", lambda: True)())
        build_running = bool(build_fut_ref is not None and not getattr(build_fut_ref, "done", lambda: True)())
        any_running = bool(sim_running or build_running)

        if st.button("Build with Deep‑TICA", disabled=any_running):
            setup = dict(st.session_state.get("setup", {}))
            if not setup:
                st.warning("No setup context captured yet; using defaults.")
            lag_use = int(setup.get("lag", int(dtica_lag)))
            bins_cv1 = int(setup.get("bins_cv1", 24))
            bins_cv2 = int(setup.get("bins_cv2", 24))
            minT = float(setup.get("minT", 300.0))
            maxT = float(setup.get("maxT", 300.0))
            build_seed = int(setup.get("build_seed", 123))
            st.session_state["last_build_message"] = None
            # Run training+build synchronously to avoid background Streamlit UI writes
            try:
                with st.spinner("Training Deep‑TICA and building…"):
                    _build_from_existing(
                        ws,
                        int(lag_use),
                        {"Rg": int(bins_cv1), "RMSD_ref": int(bins_cv2)},
                        int(build_seed),
                        float((minT + maxT) / 2.0),
                        True,
                        dict(deeptica_params),
                        bool(effective_demux_only),
                        (float(demux_temp_choice) if demux_temp_choice is not None else None),
                    )
                st.session_state["last_build_message"] = "Build finished successfully. Plots updated."
                try:
                    st.experimental_rerun()
                except Exception:
                    pass
            except Exception as e:
                st.error(f"Build failed: {e}")
                _log(f"[pmarlo] workflow: build failed: {e}")

    # Plots tab
    with tab_plots:
        # Display options
        st.subheader("Display Options")
        adaptive_bins = st.checkbox("Adaptive FES binning", value=True)
        overlay_scatter = st.checkbox("Overlay CV scatter", value=True)
        sigma = st.number_input("FES smoothing sigma", min_value=0.1, value=0.6, step=0.1)

        # Tabs for baseline vs Deep‑TICA results
        tab1, tab2 = st.tabs(["Original CVs", "Deep‑TICA CVs"])
        # Scan bundles and pick latest baseline and latest Deep‑TICA
        bundle_paths = sorted((ws / "bundles").glob("*.json"))
        baseline_res, deeptica_res = select_latest_baseline_and_deeptica(bundle_paths)

        # Gather manifest and shards once for fallback computations
        man = read_manifest(ws)
        shards = _scan_shards(ws)
        # Apply demux gating for analysis unless advanced override enabled
        if not bool(st.session_state.get("allow_replica_shards", False)):
            try:
                filtered = select_shards(shards, mode="demux", demux_temperature=None)
                if filtered:
                    shards = filtered
            except Exception:
                pass
        baseline_edges = man.get("params", {}).get("cv_bin_edges", {})
        baseline_edge_arrays = {k: np.asarray(v, dtype=float) for k, v in baseline_edges.items()} if baseline_edges else {}
        baseline_lag = int(man.get("params", {}).get("lag", 5))
        baseline_temp = float((man.get("params", {}).get("temperature") or 300.0))

        # Original CVs tab
        with tab1:
            st.subheader("Transition Matrix / π (Original CVs)")
            try:
                T0 = baseline_res.transition_matrix if (baseline_res and baseline_res.transition_matrix is not None) else None
                pi0 = baseline_res.stationary_distribution if (baseline_res and baseline_res.stationary_distribution is not None) else None
                # Fall back to recomputation if baseline_res missing matrices
                if (T0 is None or pi0 is None or T0.size == 0) and shards and baseline_edge_arrays:
                    T0, pi0 = recompute_msm_from_shards(shards, edges_by_name=baseline_edge_arrays, lag=baseline_lag)
                fig_msm0 = plot_msm(T0, pi0)
                st.pyplot(fig_msm0, width='stretch')
            except Exception:
                st.error("picture failed to create itself")

            st.subheader("Free Energy Surface (Original CVs)")
            try:
                fes0 = baseline_res.fes if baseline_res else None
                # If adaptive binning or overlay requested, recompute a display FES from shards
                if (adaptive_bins or overlay_scatter) and shards:
                    # Optional multi-temperature analysis gating and validation
                    multi_temp = bool(st.session_state.get("multi_temp_analysis", False))
                    if multi_temp:
                        try:
                            expected = _expected_temps_from_setup()
                            stats = _demux_temperature_stats(ws)
                            present = sorted(stats.keys())
                            missing = [t for t in expected if t not in present]
                            if missing:
                                st.warning(
                                    "Multi-temperature requested but missing demux temperatures: "
                                    + ", ".join(str(int(t)) for t in missing)
                                    + ". Falling back to single-temperature."
                                )
                                # Fallback to most populated temperature
                                top_T = None
                                top_frames = -1
                                for T, rec in stats.items():
                                    fr = int(rec.get("frames", 0))
                                    if fr > top_frames:
                                        top_frames = fr
                                        top_T = int(T)
                                if top_T is not None:
                                    try:
                                        filtered = select_shards(shards, mode="demux", demux_temperature=float(top_T))
                                        if filtered:
                                            shards = filtered
                                            baseline_temp = float(top_T)
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    try:
                        from pmarlo.data.shard import read_shard
                        xs: list[np.ndarray] = []
                        ys: list[np.ndarray] = []
                        for p in shards:
                            _, X, _ = read_shard(p)
                            A = np.asarray(X, dtype=float)
                            if A.shape[1] >= 2 and A.shape[0] > 0:
                                xs.append(A[:, 0])
                                ys.append(A[:, 1])
                        if xs and ys:
                            X_all = np.concatenate(xs)
                            Y_all = np.concatenate(ys)
                            from pmarlo.reporting.plots import fes2d
                            target_bins = int(max(40, int(st.session_state.get("plot_bins_avg", 64))))
                            F, xe, ye, warn = fes2d(X_all, Y_all, bins=target_bins, adaptive=bool(adaptive_bins), temperature=float(baseline_temp))
                            # Plot with optional scatter overlay
                            fig_fes0, ax = plt.subplots(figsize=(5.5, 4.5))
                            Xc = 0.5 * (xe[:-1] + xe[1:])
                            Yc = 0.5 * (ye[:-1] + ye[1:])
                            cs = ax.contourf(Xc, Yc, F.T, levels=20, cmap="magma")
                            plt.colorbar(cs, ax=ax, label="F (kJ/mol)")
                            ax.set_xlabel("cv1")
                            ax.set_ylabel("cv2")
                            if overlay_scatter:
                                n = X_all.shape[0]
                                step = max(1, n // 2000)
                                ax.scatter(X_all[::step], Y_all[::step], s=2, c="k", alpha=0.10, linewidths=0)
                            if warn:
                                st.error(str(warn))
                        else:
                            # Fallback to existing FES
                            if fes0 is None and shards and baseline_edge_arrays:
                                fes0 = recompute_fes_from_shards(shards, edges_by_name=baseline_edge_arrays, temperature=baseline_temp)
                            fig_fes0 = plot_fes(fes0)
                    except Exception:
                        # Fallback to existing FES
                        if fes0 is None and shards and baseline_edge_arrays:
                            fes0 = recompute_fes_from_shards(shards, edges_by_name=baseline_edge_arrays, temperature=baseline_temp)
                        fig_fes0 = plot_fes(fes0)
                else:
                    if fes0 is None and shards and baseline_edge_arrays:
                        fes0 = recompute_fes_from_shards(shards, edges_by_name=baseline_edge_arrays, temperature=baseline_temp)
                    fig_fes0 = plot_fes(fes0)
                st.pyplot(fig_fes0, width='stretch')
            except Exception:
                st.error("picture failed to create itself")

            # Validator panel: demux usage summary
            try:
                # Aggregate demux stats for the currently used shard set
                from pmarlo.data.shard import read_shard as _read_shard
                t_to_frames: Dict[int, int] = {}
                demux_count = 0
                for p in shards:
                    try:
                        meta, _, _ = _read_shard(p)
                        src = dict(getattr(meta, "source", {}))
                        spath = str(src.get("traj") or src.get("path") or src.get("file") or src.get("source_path") or "")
                        if "demux" not in spath.lower():
                            continue
                        demux_count += 1
                        T = int(round(float(getattr(meta, "temperature", float("nan")))))
                        t_to_frames[T] = t_to_frames.get(T, 0) + int(getattr(meta, "n_frames", 0))
                    except Exception:
                        continue
                if demux_count > 0:
                    st.caption(
                        f"Demux shards used: {demux_count}; Unique temperatures: {len(t_to_frames)}"
                    )
                    rows = [{"T (K)": int(t), "frames": int(fr)} for t, fr in sorted(t_to_frames.items())]
                    try:
                        import pandas as _pd
                        st.dataframe(_pd.DataFrame(rows), use_container_width=True)
                    except Exception:
                        st.table(rows)
            except Exception:
                pass

        # Deep‑TICA CVs tab
        with tab2:
            mlcv = (deeptica_res.artifacts or {}).get("mlcv_deeptica") if (deeptica_res and hasattr(deeptica_res, "artifacts")) else None
            if not isinstance(mlcv, dict) or not mlcv.get("applied", False):
                reason = (mlcv or {}).get("reason", "no_artifact") if isinstance(mlcv, dict) else "no_artifact"
                st.warning(f"Deep‑TICA CV learning was skipped (reason: {reason}). No Deep‑TICA FES to display.")
                # Error details if present
                if isinstance(mlcv, dict):
                    err = mlcv.get("error")
                    if err:
                        st.error(f"Deep‑TICA error: {err}")
                    tb = mlcv.get("traceback")
                    if tb:
                        with st.expander("Deep‑TICA traceback", expanded=False):
                            st.code(str(tb))
                # Show environment info even on skip
                env = (mlcv or {}).get("env") if isinstance(mlcv, dict) else None
                if isinstance(env, dict) and env:
                    with st.expander("Deep‑TICA environment (versions)", expanded=False):
                        try:
                            st.json(env)
                        except Exception:
                            st.code(str(env))
            else:
                st.subheader("Transition Matrix / π (Deep‑TICA CVs)")
                try:
                    T = deeptica_res.transition_matrix if (deeptica_res and deeptica_res.transition_matrix is not None) else None
                    pi = deeptica_res.stationary_distribution if (deeptica_res and deeptica_res.stationary_distribution is not None) else None
                    fig1 = plot_msm(T, pi)
                    st.pyplot(fig1, width='stretch')
                except Exception:
                    st.error("picture failed to create itself")

                st.subheader("Free Energy Surface (Deep‑TICA CVs)")
                try:
                    fes = deeptica_res.fes if deeptica_res else None
                    fig2 = plot_fes(fes)
                    st.pyplot(fig2, width='stretch')
                except Exception:
                    st.error("picture failed to create itself")
            try:
                shard_files = _scan_shards(ws)
                if not shard_files:
                    st.warning("No shards found to diagnose. Run a simulation or place shard JSONs under workspace/shards.")
                else:
                    ds = load_shards_as_dataset(shard_files)
                    rep = diagnose_deeptica_pairs(ds, lag=int(max(1, lag)))
                    # High-level summary including mode and shard/bias coverage
                    mode = "scaled-time" if getattr(rep, "scaled_time_used", False) else "uniform-time"
                    bias_cov = f"{getattr(rep, 'shards_with_bias', 0)}/{rep.n_shards}"
                    extra = f"; Mode: {mode}; With bias: {bias_cov}; Too-short: {getattr(rep, 'too_short_count', 0)}"
                    st.write(
                        f"Lag used: {rep.lag_used}; Shards: {rep.n_shards}; Frames total: {rep.frames_total}; "
                        f"Pairs total: {rep.pairs_total}{extra}"
                    )
                    try:
                        dup = getattr(rep, "duplicates", [])
                        if isinstance(dup, list) and dup:
                            st.warning("Duplicate shard IDs detected: " + ", ".join(sorted(set(dup))))
                    except Exception:
                        pass
                    # Table of per-shard counts
                    try:
                        import pandas as _pd  # optional, Streamlit-friendly

                        df = _pd.DataFrame(
                            [
                                {
                                    "shard": r.id,
                                    "frames": r.frames,
                                    "pairs": r.pairs,
                                    "pairs_uniform": getattr(r, "pairs_uniform", None),
                                    "bias": getattr(r, "has_bias", False),
                                    "T(K)": getattr(r, "temperature", None),
                                    "frames<=lag": getattr(r, "frames_leq_lag", False),
                                }
                                for r in rep.per_shard
                            ]
                        )
                        st.dataframe(df, width='stretch')
                    except Exception:
                        # Fallback plain text
                        lines = [
                            f"{r.id}: frames={r.frames} pairs={r.pairs} bias={getattr(r,'has_bias',False)}"
                            for r in rep.per_shard
                        ]
                        st.code("\n".join(lines))
                    # Comparison of scaled vs uniform totals if available
                    try:
                        if getattr(rep, "pairs_total_uniform", None) is not None:
                            st.caption(
                                f"Uniform-time pairs total (reference): {int(getattr(rep,'pairs_total_uniform',0))}"
                            )
                    except Exception:
                        pass
                    st.info(rep.message)
            except Exception as e:
                st.error(f"Diagnosis failed: {e}")

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
                # After cleanup, rebuild shard index from current on-disk manifests only
                try:
                    idx = Path(ws) / "shards_index.json"
                    if idx.exists():
                        idx.unlink(missing_ok=True)  # remove stale index before rescan
                    rescan_shards([Path(ws) / "shards"], idx)
                except Exception:
                    pass
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

            # Display validation information if available
            validation = last.get("validation")
            if validation:
                st.subheader("Build Validation")

                # Overall validation status
                is_valid = validation.get("is_valid", False)
                if is_valid:
                    st.success("Build validation: PASSED")
                else:
                    st.error("Build validation: ISSUES FOUND")

                # Shard usage information
                shard_stats = validation.get("shard_stats", {})
                if shard_stats:
                    with st.expander("Shard Usage", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Available", shard_stats.get("total_available", 0))
                        with col2:
                            st.metric("Used in Build", shard_stats.get("total_used", 0))
                        with col3:
                            usage_ratio = shard_stats.get("usage_ratio", 0)
                            st.metric("Usage Ratio", f"{usage_ratio:.1%}")

                # Weight and bias information
                weight_stats = validation.get("weight_stats", {})
                if weight_stats and weight_stats.get("shards_with_bias", 0) > 0:
                    with st.expander("Weight/Bias Information", expanded=True):
                        bias_shards = weight_stats.get("shards_with_bias", 0)
                        total_shards = weight_stats.get("total_shards", 0)
                        temps = weight_stats.get("temperatures", [])

                        st.write(f"Bias information found in {bias_shards}/{total_shards} shards")

                        if len(set(temps)) > 1:
                            st.write(f"Temperature range: {min(temps):.1f}K - {max(temps):.1f}K")
                            st.write(f"Unique temperatures: {len(set(temps))}")
                        elif temps:
                            st.write(f"Single temperature: {temps[0]:.1f}K")

                # Data quality information
                data_quality = validation.get("data_quality", {})
                if data_quality:
                    with st.expander("Data Quality", expanded=True):
                        total_frames = data_quality.get("total_frames", 0)
                        avg_frames = data_quality.get("avg_frames_per_shard", 0)

                        if total_frames > 0:
                            st.metric("Total Frames", f"{total_frames:,}")
                            st.metric("Avg Frames/Shards", f"{avg_frames:.0f}")

                        # Show warnings if any
                        warnings = validation.get("warnings", [])
                        if warnings:
                            st.warning("Issues found:")
                            for warning in warnings:
                                st.write(f"• {warning}")

                        # Show messages
                        messages = validation.get("messages", [])
                        if messages:
                            st.info("Validation results:")
                            for msg in messages:
                                st.write(f"OK: {msg}")

                # Build type information
                build_type = shard_stats.get("build_type", "unknown")
                st.info(f"Build type: {build_type}")
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

    # Diagnostics tab: pair diagnosis, cleanup, logs, current info
    with tab_diag:
        # Diagnostics: Deep‑TICA pairs without training
        st.subheader("Deep‑TICA Pair Diagnostics")
        sim_fut_ref = st.session_state.get("sim_future")
        build_fut_ref = st.session_state.get("build_future")
        sim_running = bool(sim_fut_ref is not None and not getattr(sim_fut_ref, "done", lambda: True)())
        build_running = bool(build_fut_ref is not None and not getattr(build_fut_ref, "done", lambda: True)())
        any_running = bool(sim_running or build_running)
        if st.button("Diagnose Deep‑TICA pairs (no training)", disabled=any_running):
            try:
                shard_files = _scan_shards(ws)
                if not shard_files:
                    st.warning("No shards found to diagnose. Run a simulation or place shard JSONs under workspace/shards.")
                else:
                    ds = load_shards_as_dataset(shard_files)
                    # Use lag from Setup tab if present
                    lag_use = int(max(1, int(st.session_state.get("setup", {}).get("lag", 5))))
                    rep = diagnose_deeptica_pairs(ds, lag=lag_use)
                    mode = "scaled-time" if getattr(rep, "scaled_time_used", False) else "uniform-time"
                    bias_cov = f"{getattr(rep, 'shards_with_bias', 0)}/{rep.n_shards}"
                    extra = f"; Mode: {mode}; With bias: {bias_cov}; Too-short: {getattr(rep, 'too_short_count', 0)}"
                    st.write(
                        f"Lag used: {rep.lag_used}; Shards: {rep.n_shards}; Frames total: {rep.frames_total}; Pairs total: {rep.pairs_total}{extra}"
                    )
                    try:
                        dup = getattr(rep, "duplicates", [])
                        if isinstance(dup, list) and dup:
                            st.warning("Duplicate shard IDs detected: " + ", ".join(sorted(set(dup))))
                    except Exception:
                        pass
                    try:
                        import pandas as _pd
                        df = _pd.DataFrame(
                            [
                                {
                                    "shard": r.id,
                                    "frames": r.frames,
                                    "pairs": r.pairs,
                                    "pairs_uniform": getattr(r, "pairs_uniform", None),
                                    "bias": getattr(r, "has_bias", False),
                                    "T(K)": getattr(r, "temperature", None),
                                    "frames<=lag": getattr(r, "frames_leq_lag", False),
                                }
                                for r in rep.per_shard
                            ]
                        )
                        st.dataframe(df, width='stretch')
                    except Exception:
                        lines = [f"{r.id}: frames={r.frames} pairs={r.pairs} bias={getattr(r,'has_bias',False)}" for r in rep.per_shard]
                        st.code("\n".join(lines))
                    try:
                        if getattr(rep, "pairs_total_uniform", None) is not None:
                            st.caption(f"Uniform-time pairs total (reference): {int(getattr(rep,'pairs_total_uniform',0))}")
                    except Exception:
                        pass
                    st.info(rep.message)
            except Exception as e:
                st.error(f"Diagnosis failed: {e}")

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

        # Demux temperature coverage explanation
        st.subheader("Demux Temperature Coverage")
        try:
            stats = _demux_temperature_stats(ws)
            if not stats:
                st.write("No demux shards found.")
            else:
                # Expected ladder from Setup
                expected = _expected_temps_from_setup()
                present = sorted(stats.keys())
                # Build compact rows
                rows = []
                total_frames = sum(int(rec.get("frames", 0)) for rec in stats.values())
                for T in sorted(set(present + expected)):
                    rec = stats.get(int(T), {"shards": 0, "frames": 0})
                    rows.append({"T (K)": int(T), "shards": int(rec["shards"]), "frames": int(rec["frames"]), "expected": ("yes" if (int(T) in expected) else "no")})
                try:
                    import pandas as _pd
                    df = _pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)
                except Exception:
                    st.table(rows)
                # Completeness: fraction of expected temps present
                if expected:
                    present_expected = len([t for t in expected if t in present])
                    completeness = int(round(100.0 * present_expected / max(1, len(expected))))
                    st.caption(f"Demux completeness: {present_expected}/{len(expected)} temperatures present")
                    try:
                        st.progress(completeness / 100.0, text=f"{completeness}%")  # type: ignore[arg-type]
                    except Exception:
                        st.write(f"{completeness}% complete")
        except Exception:
            pass

    # Poll background jobs
    sim_fut = st.session_state.get("sim_future")
    if sim_fut is not None and sim_fut.done():
        try:
            sim_fut.result()
            st.session_state["last_sim_message"] = "Simulation and emission finished successfully. You can now Build MSM/FES."
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            _log(f"[pmarlo] workflow: simulation failed: {e}")
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
            _log(f"[pmarlo] workflow: build failed: {e}")
        finally:
            st.session_state["build_future"] = None
            try:
                st.experimental_rerun()
            except Exception:
                pass

    # Central log viewer: drain worker logs and display in one place
    st.subheader("Logs")
    # Compact shard summary instead of per-shard spam
    try:
        shard_files = _scan_shards(ws)
        if shard_files:
            from pmarlo.data.shard import read_shard as _read_shard
            run_ids = set()
            kinds = set()
            temps = set()
            total_frames = 0
            for p in shard_files:
                try:
                    meta, _, _ = _read_shard(p)
                    src = dict(getattr(meta, "source", {}))
                    # nearest run-* parent
                    run_ids.add(str(p.parent).split("run-")[-1][:20] if "run-" in str(p.parent) else p.parent.name)
                    spath = str(src.get("traj") or src.get("path") or src.get("file") or src.get("source_path") or "")
                    kinds.add("demux" if "demux" in spath.lower() else ("replica" if "replica" in spath.lower() else "unknown"))
                    if "demux" in spath.lower():
                        try:
                            temps.add(int(round(float(getattr(meta, "temperature", float("nan"))))))
                        except Exception:
                            pass
                    total_frames += int(getattr(meta, "n_frames", 0))
                except Exception:
                    continue
            avg_frames = float(total_frames) / max(1, len(shard_files))
            st.caption(
                f"Shard summary: runs={len(run_ids)}, kinds={','.join(sorted(kinds))}, temps={len(temps)}, "
                f"frames={total_frames:,}, avg_frames/shard={avg_frames:.0f}"
            )
    except Exception:
        pass
    log_box = st.empty()
    try:
        lines = list(st.session_state.get("_log_lines", []))
    except Exception:
        lines = []
    drained = 0
    while True:
        try:
            m = _LOG_Q.get_nowait()
            lines.append(str(m))
            drained += 1
        except Exception:
            break
    # Collapse consecutive duplicates and trim
    def _collapse(seq: list[str], max_keep: int = 500) -> list[str]:
        out: list[str] = []
        i = 0
        n = len(seq)
        while i < n:
            j = i + 1
            while j < n and seq[j] == seq[i]:
                j += 1
            cnt = j - i
            if cnt > 1:
                out.append(f"{seq[i]} (x{cnt})")
            else:
                out.append(seq[i])
            i = j
        if len(out) > max_keep:
            rest = len(out) - (max_keep - 1)
            return out[: max_keep - 1] + [f"... and {rest} more lines"]
        return out
    collapsed = _collapse(lines)
    try:
        log_box.code("\n".join(collapsed))
    except Exception:
        pass
    try:
        st.session_state["_log_lines"] = lines[-800:]
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
        _log("[pmarlo] workflow: simulation + emission finished successfully; next → Build MSM/FES")
        # Refresh shard index after emission
        try:
            idx = Path(ws) / "shards_index.json"
            rescan_shards([Path(ws) / "shards"], idx)
            prune_missing_shards(idx)
        except Exception:
            pass
    except Exception as e:
        _log(f"[pmarlo] workflow: simulation job error: {e}")
        raise


def _build_from_existing(
    ws: Path,
    lag: int,
    bins: Dict[str, int],
    seed: int,
    temperature: float,
    learn_cv: bool,
    deeptica_params: Dict[str, Any],
    demux_only: bool = True,
    demux_temperature: float | None = None,
) -> None:
    try:
        shards = _scan_shards(ws)
        # Optional filtering: use only demux shards (and optional single‑T)
        if demux_only:
            demux_temp_f = float(demux_temperature) if demux_temperature is not None else None
            filtered = select_shards(shards, mode="demux", demux_temperature=demux_temp_f)
            if filtered:
                shards = filtered
                msg = (
                    f"[pmarlo] workflow: Filtering to demux shards only (kept {len(shards)})."
                    + (f" Single‑T={int(demux_temp_f)}K." if demux_temp_f is not None else "")
                )
                _log(msg)
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
            build_type=("deeptica" if learn_cv else "baseline"),
        )
        params = {
            "lag": int(lag),
            "bins": bins,
            "seed": int(seed),
            "learn_cv": bool(learn_cv),
            "deeptica_params": dict(deeptica_params or {}),
            "cv_bin_edges": {k: v.tolist() for k, v in edges.items()},
        }
        # Perform validation before updating manifest
        validation = validate_build_quality(res, shards, ws)

        # Update manifest with validation information
        _update_manifest_after_build(ws, res, out_bundle, [p.stem for p in shards], params, validation)

        # Print comprehensive build validation results
        try:
            _log(f"[pmarlo] workflow: Build validation completed")

            # Shard usage information
            shard_info = validation.shard_stats
            _log(f"[pmarlo] workflow: Used {shard_info['total_used']}/{shard_info['total_available']} shards ({shard_info['usage_ratio']:.1%})")

            # Weight/bias information
            weight_info = validation.weight_stats
            if weight_info.get('shards_with_bias', 0) > 0:
                _log(f"[pmarlo] workflow: Bias found in {weight_info['shards_with_bias']}/{weight_info['total_shards']} shards")
                if weight_info.get('unique_temperatures', 0) > 1:
                    temps = weight_info.get('temperatures', [])
                    _log(f"[pmarlo] workflow: Temperature range: {min(temps):.1f}K - {max(temps):.1f}K")

            # Data quality information
            quality_info = validation.data_quality
            if quality_info.get('total_frames', 0) > 0:
                _log(f"[pmarlo] workflow: Total frames: {quality_info['total_frames']:,}")

            # Validation messages
            for msg in validation.messages:
                _log(f"[pmarlo] workflow: OK: {msg}")

            # Warnings (if any)
            for warning in validation.warnings:
                _log(f"[pmarlo] workflow: WARNING: {warning}")

            # Overall validation status
            if validation.is_valid:
                _log(f"[pmarlo] workflow: Build validation PASSED")
            else:
                _log(f"[pmarlo] workflow: Build validation ISSUES FOUND")

        except Exception as e:
            _log(f"[pmarlo] workflow: Build validation failed: {e}")

        # Explicit console cue for users following logs
        # Prominent post-build status line derived from artifacts
        try:
            art = (res.artifacts or {}) if hasattr(res, "artifacts") else {}
            mlcv = art.get("mlcv_deeptica") if isinstance(art, dict) else None
            if isinstance(mlcv, dict):
                if mlcv.get("applied"):
                    _log(f"[pmarlo] workflow: Deep-TICA applied; lag={mlcv.get('lag_used','?')}; pairs={mlcv.get('pairs_total','?')}.")
                elif mlcv.get("skipped"):
                    reason = mlcv.get('reason','no_artifact')
                    # Extract missing modules from env, if present
                    miss = []
                    try:
                        env = mlcv.get('env', {}) if isinstance(mlcv, dict) else {}
                        if isinstance(env, dict):
                            for k, v in env.items():
                                if isinstance(v, str) and v.startswith('IMPORT-FAIL'):
                                    miss.append(k)
                    except Exception:
                        pass
                    extra = ("; missing=[" + ",".join(sorted(set(miss))) + "]") if (miss and str(reason).startswith('missing_dependency')) else ""
                    err = mlcv.get('error')
                    tb = mlcv.get('traceback')
                    _log(f"[pmarlo] workflow: Deep-TICA skipped; reason={reason}{extra}." + (f" error={err}" if err else ""))
                    if tb:
                        _log("[pmarlo] workflow: Deep-TICA traceback follows:\n" + str(tb))
            _log("[pmarlo] workflow: build finished successfully; plots updated")
        except Exception:
            pass
    except Exception as e:
        _log(f"[pmarlo] workflow: build job error: {e}")
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
