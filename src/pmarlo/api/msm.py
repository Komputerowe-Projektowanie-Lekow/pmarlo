from __future__ import annotations

import logging
from typing import Literal, Optional, Sequence, List
from pathlib import Path

import numpy as np

from pmarlo.markov_state_model import MarkovStateModel
from pmarlo.markov_state_model._msm_utils import (
    candidate_lag_ladder,
    build_simple_msm,
    pcca_like_macrostates,
    compute_macro_populations,
    lump_micro_to_macro_T,
    compute_macro_mfpt,
)
from pmarlo.markov_state_model.ck_runner import run_ck
from pmarlo.markov_state_model.free_energy import generate_1d_pmf
from pmarlo.api.features import compute_universal_embedding
from pmarlo.api.fes import generate_free_energy_surface
from pmarlo.reporting.plots import save_pmf_line, save_fes_contour
from pmarlo.utils.path_utils import ensure_directory

logger = logging.getLogger("pmarlo")


def analyze_msm(  # noqa: C901
    trajectory_files: List[str],
    topology_pdb: str | Path,
    output_dir: str | Path,
    feature_type: str = "phi_psi",
    analysis_temperatures: Optional[List[float]] = None,
    use_effective_for_uncertainty: bool = True,
    use_tica: bool = True,
    random_state: int | None = 42,
    traj_stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
) -> Path:
    """Build and analyze an MSM, saving plots and artifacts.

    Parameters
    ----------
    trajectory_files:
        Trajectory file paths.
    topology_pdb:
        Topology in PDB format.
    output_dir:
        Destination directory.
    feature_type:
        Feature specification string.
    analysis_temperatures:
        Optional list of temperatures for analysis.
    use_effective_for_uncertainty:
        Whether to use effective counts for uncertainty.
    use_tica:
        Whether to apply TICA reduction.
    random_state:
        Seed for deterministic clustering. ``None`` uses the global state.
    traj_stride:
        Stride for loading trajectory frames.
    atom_selection:
        MDTraj atom selection string or explicit atom indices used when
        loading trajectories.
    chunk_size:
        Number of frames per chunk when streaming trajectories from disk.

    Returns
    -------
    Path
        The analysis output directory.
    """
    logger.info(
        "[msm] Starting MSM analysis: n_trajectories=%d, feature_type=%s, use_tica=%s",
        len(trajectory_files),
        feature_type,
        use_tica,
    )
    logger.debug(
        "[msm] Parameters: topology=%s, output_dir=%s, stride=%d, chunk_size=%d",
        topology_pdb,
        output_dir,
        traj_stride,
        chunk_size,
    )

    msm_out = Path(output_dir) / "msm_analysis"
    logger.debug("[msm] MSM output directory: %s", msm_out)

    logger.info("[msm] Initializing MarkovStateModel...")
    msm = MarkovStateModel(
        trajectory_files=trajectory_files,
        topology_file=str(topology_pdb),
        temperatures=analysis_temperatures or [300.0],
        output_dir=str(msm_out),
        random_state=random_state,
    )
    logger.debug("[msm] MarkovStateModel initialized with temperatures: %s", analysis_temperatures or [300.0])

    # Configure MSM parameters
    if use_effective_for_uncertainty and hasattr(msm, "count_mode"):
        msm.count_mode = "sliding"  # type: ignore[attr-defined]
        logger.debug("[msm] Set count_mode to 'sliding' for effective counts")

    # Load trajectories
    logger.info("[msm] Loading trajectories with stride=%d...", traj_stride)
    if hasattr(msm, "load_trajectories"):
        msm.load_trajectories(  # type: ignore[attr-defined]
            stride=traj_stride, atom_selection=atom_selection, chunk_size=chunk_size
        )
        logger.info("[msm] Trajectories loaded successfully")
    else:
        logger.warning("[msm] MSM object does not support load_trajectories method")

    # Compute features
    ft = feature_type
    if use_tica and ("tica" not in feature_type.lower()):
        ft = f"{feature_type}_tica"
        logger.debug("[msm] Modified feature_type to include TICA: %s", ft)

    logger.info("[msm] Computing features: %s", ft)
    if hasattr(msm, "compute_features"):
        msm.compute_features(feature_type=ft)  # type: ignore[attr-defined]
        logger.info("[msm] Features computed successfully")
    else:
        logger.warning("[msm] MSM object does not support compute_features method")

    # Cluster
    N_CLUSTERS = 8
    logger.info("[msm] Clustering features into %d states...", N_CLUSTERS)
    if hasattr(msm, "cluster_features"):
        msm.cluster_features(n_states=int(N_CLUSTERS))  # type: ignore[attr-defined]
        logger.info("[msm] Clustering complete")
    else:
        logger.warning("[msm] MSM object does not support cluster_features method")

    # Method selection
    method = (
        "tram"
        if analysis_temperatures
        and len(analysis_temperatures) > 1
        and len(trajectory_files) > 1
        else "standard"
    )
    logger.info("[msm] Selected MSM estimation method: %s", method)

    # ITS and lag selection
    logger.debug("[msm] Calculating adaptive lag time parameters...")
    try:
        total_frames = sum(
            getattr(t, "n_frames", 0) for t in getattr(msm, "trajectories", [])
        )
    except Exception:
        total_frames = 0
        logger.warning("[msm] Could not determine total frames")

    max_lag = 250
    try:
        if total_frames > 0:
            max_lag = int(min(500, max(150, total_frames // 5)))
            logger.debug("[msm] Calculated max_lag=%d from %d total frames", max_lag, total_frames)
    except Exception:
        max_lag = 250
        logger.warning("[msm] Using default max_lag=%d", max_lag)

    candidate_lags = candidate_lag_ladder(min_lag=1, max_lag=max_lag)
    logger.info("[msm] Generated %d candidate lag times (1 to %d)", len(candidate_lags), max_lag)

    logger.info("[msm] Building initial MSM with lag_time=5...")
    if hasattr(msm, "build_msm"):
        msm.build_msm(lag_time=5, method=method)  # type: ignore[attr-defined]
        logger.debug("[msm] Initial MSM built")

    logger.info("[msm] Computing implied timescales...")
    if hasattr(msm, "compute_implied_timescales"):
        msm.compute_implied_timescales(lag_times=candidate_lags, n_timescales=3)  # type: ignore[attr-defined]
        logger.info("[msm] Implied timescales computed for %d lag times", len(candidate_lags))

    chosen_lag = 10
    logger.debug("[msm] Selecting optimal lag time...")
    try:
        its_data = getattr(msm, "implied_timescales", None)
        if its_data is not None and hasattr(its_data, "__getitem__"):
            lags = np.array(its_data["lag_times"])  # type: ignore[index]
            its = np.array(its_data["timescales"])  # type: ignore[index]
            logger.debug("[msm] Using computed ITS data for lag selection")
        else:
            lags = np.array(candidate_lags)
            its = np.ones((len(candidate_lags), 3)) * 10.0
            logger.debug("[msm] Using default ITS values")

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
            logger.info("[msm] Selected optimal lag_time=%d", chosen_lag)
    except Exception as e:
        chosen_lag = 10
        logger.warning("[msm] Lag time selection failed, using default lag_time=%d: %s", chosen_lag, e)

    logger.info("[msm] Building final MSM with chosen lag_time=%d...", chosen_lag)
    if hasattr(msm, "build_msm"):
        msm.build_msm(lag_time=chosen_lag, method=method)  # type: ignore[attr-defined]
        logger.info("[msm] Final MSM built successfully")

    # CK test with macro → micro fallback
    logger.debug("[msm] Running Chapman-Kolmogorov test...")
    try:
        dtrajs = getattr(msm, "dtrajs", None)
        lag_time = getattr(msm, "lag_time", chosen_lag)
        output_dir_ck = getattr(msm, "output_dir", output_dir)
        if dtrajs is not None:
            run_ck(dtrajs, lag_time, output_dir_ck, macro_k=3)
            logger.info("[msm] Chapman-Kolmogorov test completed")
        else:
            logger.warning("[msm] No dtrajs available for CK test")
    except Exception as e:
        logger.warning("[msm] Chapman-Kolmogorov test failed: %s", e)

    try:
        total_frames_fes = sum(
            getattr(t, "n_frames", 0) for t in getattr(msm, "trajectories", [])
        )
    except Exception:
        total_frames_fes = 0
    adaptive_bins = max(20, min(50, int((total_frames_fes or 0) ** 0.5))) or 20
    logger.debug("[msm] Using adaptive_bins=%d for FES plots", adaptive_bins)

    # Plot FES/PMF based on feature_type
    if feature_type.lower().startswith("universal"):
        logger.info("[msm] Generating universal embedding-based FES plots...")
        try:
            # Build one universal embedding and reuse for PMF(1D) and FES(2D)
            traj_all = None
            trajectories = getattr(msm, "trajectories", [])
            for t in trajectories:
                traj_all = t if traj_all is None else traj_all.join(t)
            if traj_all is not None:
                # Choose method with Literal-typed variable for mypy
                if "vamp" in feature_type.lower():
                    red_method: Literal["vamp", "tica", "pca"] = "vamp"
                elif "tica" in feature_type.lower():
                    red_method = "tica"
                else:
                    red_method = "pca"
                logger.debug("[msm] Selected reduction method: %s", red_method)

                # Reuse cached features for the concatenated trajectory as well
                cache_dir = (
                    Path(str(getattr(msm, "output_dir", output_dir))) / "feature_cache"
                )
                ensure_directory(cache_dir)
                logger.debug("[msm] Feature cache directory: %s", cache_dir)

                logger.info("[msm] Computing universal embedding (n_components=2)...")
                Y2, _ = compute_universal_embedding(
                    traj_all,
                    feature_specs=None,
                    align=True,
                    method=red_method,
                    lag=int(max(1, getattr(msm, "lag_time", None) or 10)),
                    n_components=2,
                    cache_path=str(cache_dir),
                )
                logger.info("[msm] Universal embedding computed: shape=%s", Y2.shape)

                # 1) PMF on IC1
                logger.info("[msm] Generating 1D PMF on IC1...")
                pmf = generate_1d_pmf(
                    Y2[:, 0], bins=int(max(30, adaptive_bins)), temperature=300.0
                )
                _ = save_pmf_line(
                    pmf.F,
                    pmf.edges,
                    xlabel="universal IC1",
                    output_dir=str(getattr(msm, "output_dir", output_dir)),
                    filename="pmf_universal_ic1.png",
                )
                logger.info("[msm] 1D PMF plot saved: pmf_universal_ic1.png")

                # 2) 2D FES on (IC1, IC2)
                logger.info("[msm] Generating 2D FES on IC1 × IC2...")
                fes2 = generate_free_energy_surface(
                    Y2[:, 0],
                    Y2[:, 1],
                    bins=(int(adaptive_bins), int(adaptive_bins)),
                    temperature=300.0,
                    periodic=(False, False),
                    smooth=True,
                    min_count=1,
                )
                _ = save_fes_contour(
                    fes2.F,
                    fes2.xedges,
                    fes2.yedges,
                    "universal IC1",
                    "universal IC2",
                    str(getattr(msm, "output_dir", output_dir)),
                    "fes_universal_ic1_vs_ic2.png",
                )
                logger.info("[msm] 2D FES contour plot saved: fes_universal_ic1_vs_ic2.png")
            else:
                logger.warning("[msm] No trajectories available for universal embedding")
        except Exception as e:
            logger.error("[msm] Failed to generate universal embedding FES: %s", e)
    else:
        logger.debug("[msm] Skipping phi/psi-specific FES (feature_type=%s)", feature_type)

    # Generate plots and analysis results with attribute checks
    logger.info("[msm] Generating analysis plots and results...")
    if hasattr(msm, "plot_implied_timescales"):
        msm.plot_implied_timescales(save_file="implied_timescales")  # type: ignore[attr-defined]
        logger.debug("[msm] Implied timescales plot generated")
    if hasattr(msm, "plot_free_energy_profile"):
        msm.plot_free_energy_profile(save_file="free_energy_profile")  # type: ignore[attr-defined]
        logger.debug("[msm] Free energy profile plot generated")
    if hasattr(msm, "create_state_table"):
        msm.create_state_table()  # type: ignore[attr-defined]
        logger.debug("[msm] State table created")
    if hasattr(msm, "extract_representative_structures"):
        msm.extract_representative_structures(save_pdb=True)  # type: ignore[attr-defined]
        logger.debug("[msm] Representative structures extracted")
    if hasattr(msm, "save_analysis_results"):
        msm.save_analysis_results()  # type: ignore[attr-defined]
        logger.debug("[msm] Analysis results saved")

    logger.info("[msm] MSM analysis complete: output_dir=%s", msm_out)
    return msm_out


def build_msm_from_labels(
    dtrajs: list[np.ndarray], n_states: Optional[int] = None, lag: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Build a Markov State Model from discrete trajectory labels.

    Parameters
    ----------
    dtrajs : list[np.ndarray]
        List of discrete trajectory arrays (state sequences).
    n_states : Optional[int]
        Number of states. If None, inferred from max label.
    lag : int
        Lag time for transition counting.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Transition matrix and stationary distribution.
    """
    logger.info("[msm] Building MSM from labels: n_trajectories=%d, lag=%d", len(dtrajs), lag)
    if n_states is not None:
        logger.debug("[msm] Using provided n_states=%d", n_states)
    else:
        logger.debug("[msm] Inferring n_states from trajectory labels")

    T, pi = build_simple_msm(dtrajs, n_states=n_states, lag=lag)
    logger.info("[msm] MSM built: transition_matrix_shape=%s, stationary_dist_shape=%s", T.shape, pi.shape)
    return T, pi


def compute_macrostates(T: np.ndarray, n_macrostates: int = 4) -> Optional[np.ndarray]:
    """Compute macrostates from transition matrix using PCCA+.

    Parameters
    ----------
    T : np.ndarray
        Transition matrix.
    n_macrostates : int
        Number of desired macrostates.

    Returns
    -------
    Optional[np.ndarray]
        Array mapping microstates to macrostates, or None if computation fails.
    """
    logger.info("[msm] Computing %d macrostates from transition matrix (shape=%s)", n_macrostates, T.shape)
    result = pcca_like_macrostates(T, n_macrostates=n_macrostates)
    if result is not None:
        logger.info("[msm] Macrostates computed successfully")
    else:
        logger.warning("[msm] Macrostate computation failed or returned None")
    return result


def macrostate_populations(
    pi_micro: np.ndarray, micro_to_macro: np.ndarray
) -> np.ndarray:
    """Compute macrostate populations from microstate stationary distribution.

    Parameters
    ----------
    pi_micro : np.ndarray
        Microstate stationary distribution.
    micro_to_macro : np.ndarray
        Mapping from microstates to macrostates.

    Returns
    -------
    np.ndarray
        Macrostate populations.
    """
    logger.debug("[msm] Computing macrostate populations: n_microstates=%d", len(pi_micro))
    result = compute_macro_populations(pi_micro, micro_to_macro)
    logger.debug("[msm] Computed populations for %d macrostates", len(result))
    return result


def macro_transition_matrix(
    T_micro: np.ndarray, pi_micro: np.ndarray, micro_to_macro: np.ndarray
) -> np.ndarray:
    """Lump microstate transition matrix to macrostate level.

    Parameters
    ----------
    T_micro : np.ndarray
        Microstate transition matrix.
    pi_micro : np.ndarray
        Microstate stationary distribution.
    micro_to_macro : np.ndarray
        Mapping from microstates to macrostates.

    Returns
    -------
    np.ndarray
        Macrostate transition matrix.
    """
    logger.debug("[msm] Lumping micro to macro transition matrix: T_micro_shape=%s", T_micro.shape)
    result = lump_micro_to_macro_T(T_micro, pi_micro, micro_to_macro)
    logger.debug("[msm] Macro transition matrix shape=%s", result.shape)
    return result


def macro_mfpt(T_macro: np.ndarray) -> np.ndarray:
    """Compute mean first passage times between macrostates.

    Parameters
    ----------
    T_macro : np.ndarray
        Macrostate transition matrix.

    Returns
    -------
    np.ndarray
        Matrix of mean first passage times.
    """
    logger.debug("[msm] Computing macro MFPTs: T_macro_shape=%s", T_macro.shape)
    result = compute_macro_mfpt(T_macro)
    logger.debug("[msm] MFPT matrix computed: shape=%s", result.shape)
    return result
