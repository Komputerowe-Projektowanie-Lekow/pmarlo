from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Sequence, cast

import numpy as np

from pmarlo.analysis import compute_diagnostics
from pmarlo.api.features import compute_universal_embedding
from pmarlo.api.fes import generate_free_energy_surface
from pmarlo.data.aggregate import load_shards_as_dataset
from pmarlo.markov_state_model import MarkovStateModel
from pmarlo.markov_state_model.enhanced_msm import EnhancedMSMProtocol
from pmarlo.utils.msm_utils import (
    build_simple_msm,
    candidate_lag_ladder,
    compute_macro_mfpt,
    compute_macro_populations,
    lump_micro_to_macro_T,
    pcca_like_macrostates,
    select_lag_from_its,
)
from pmarlo.markov_state_model.ck_its_selector import select_optimal_lag_ck_its
from pmarlo.markov_state_model.ck_runner import run_ck
from pmarlo.markov_state_model.free_energy import generate_1d_pmf
from pmarlo.markov_state_model.results import CKITSSelectionResult, ITSResult
from pmarlo.reporting.plots import save_fes_contour, save_pmf_line
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
    msm: EnhancedMSMProtocol = cast(
        EnhancedMSMProtocol,
        MarkovStateModel(
            trajectory_files=trajectory_files,
            topology_file=str(topology_pdb),
            temperatures=analysis_temperatures or [300.0],
            output_dir=str(msm_out),
            random_state=random_state,
        ),
    )
    logger.debug(
        "[msm] MarkovStateModel initialized with temperatures: %s",
        analysis_temperatures or [300.0],
    )

    # Configure MSM parameters
    if use_effective_for_uncertainty:
        msm.count_mode = "sliding"
        logger.debug("[msm] Set count_mode to 'sliding' for effective counts")

    # Load trajectories
    logger.info("[msm] Loading trajectories with stride=%d...", traj_stride)
    msm.load_trajectories(
        stride=traj_stride, atom_selection=atom_selection, chunk_size=chunk_size
    )
    logger.info("[msm] Trajectories loaded successfully")

    # Compute features
    ft = feature_type
    if use_tica and ("tica" not in feature_type.lower()):
        ft = f"{feature_type}_tica"
        logger.debug("[msm] Modified feature_type to include TICA: %s", ft)

    logger.info("[msm] Computing features: %s", ft)
    msm.compute_features(feature_type=ft)
    logger.info("[msm] Features computed successfully")

    # Cluster
    N_CLUSTERS = 8
    logger.info("[msm] Clustering features into %d states...", N_CLUSTERS)
    msm.cluster_features(n_states=int(N_CLUSTERS))
    logger.info("[msm] Clustering complete")

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
            logger.debug(
                "[msm] Calculated max_lag=%d from %d total frames",
                max_lag,
                total_frames,
            )
    except Exception:
        max_lag = 250
        logger.warning("[msm] Using default max_lag=%d", max_lag)

    candidate_lags = candidate_lag_ladder(min_lag=1, max_lag=max_lag)
    logger.info(
        "[msm] Generated %d candidate lag times (1 to %d)", len(candidate_lags), max_lag
    )

    logger.info("[msm] Building initial MSM with lag_time=5...")
    msm.build_msm(lag_time=5, method=method)
    logger.debug("[msm] Initial MSM built")

    logger.info("[msm] Computing implied timescales...")
    msm.compute_implied_timescales(lag_times=candidate_lags, n_timescales=3)
    logger.info(
        "[msm] Implied timescales computed for %d lag times", len(candidate_lags)
    )

    # Select lag time from ITS plateau detection
    logger.debug("[msm] Selecting optimal lag time from implied timescales...")
    chosen_lag = 10  # default fallback
    try:
        its_data: ITSResult | Mapping[str, Any] | None = getattr(
            msm, "implied_timescales", None
        )
        if its_data is None:
            logger.warning("[msm] No ITS data available, using default lag_time=10")
        elif isinstance(its_data, ITSResult):
            lags = np.asarray(its_data.lag_times, dtype=int)
            its = np.asarray(its_data.timescales, dtype=float)
            logger.debug(
                "[msm] ITS data extracted: lags shape=%s, its shape=%s",
                lags.shape,
                its.shape,
            )
            from pmarlo.utils.msm_utils import select_lag_from_its as detect_plateau

            chosen_lag = detect_plateau(
                lags, its, min_lag_idx=3, plateau_threshold=0.15
            )
            logger.info("[msm] Selected lag_time=%d from ITS plateau detection", chosen_lag)
        elif isinstance(its_data, Mapping):
            lag_entry = its_data.get("lag_times")
            timescales_entry = its_data.get("timescales")
            if lag_entry is None or timescales_entry is None:
                logger.warning("[msm] ITS mapping missing expected keys")
            else:
                lags = np.asarray(lag_entry, dtype=int)
                its = np.asarray(timescales_entry, dtype=float)
                logger.debug(
                    "[msm] ITS data extracted (mapping interface): lags shape=%s, its shape=%s",
                    lags.shape,
                    its.shape,
                )
                from pmarlo.utils.msm_utils import select_lag_from_its as detect_plateau

                chosen_lag = detect_plateau(
                    lags, its, min_lag_idx=3, plateau_threshold=0.15
                )
                logger.info(
                    "[msm] Selected lag_time=%d from ITS plateau detection", chosen_lag
                )
        else:
            logger.warning("[msm] ITS data exists but has unexpected structure")
    except Exception as e:
        chosen_lag = 10
        logger.warning(
            "[msm] Lag time selection from ITS failed, using default lag_time=%d: %s",
            chosen_lag,
            e,
        )

    # Store selected lag for downstream analysis
    msm.lag_time = chosen_lag

    logger.info("[msm] Building final MSM with chosen lag_time=%d...", chosen_lag)
    msm.build_msm(lag_time=chosen_lag, method=method)
    logger.info("[msm] Final MSM built successfully")

    # CK test with macro → micro fallback
    logger.debug("[msm] Running Chapman-Kolmogorov test...")
    ck_result = None
    ck_max_error = float("inf")
    ck_pass = False
    try:
        dtrajs = getattr(msm, "dtrajs", None)
        lag_time = getattr(msm, "lag_time", chosen_lag)
        output_dir_ck = getattr(msm, "output_dir", output_dir)
        if dtrajs is not None:
            ck_result = run_ck(dtrajs, lag_time, output_dir_ck, macro_k=3)
            ck_max_error = ck_result.max_error
            # Default threshold: 0.05 (5% RMS error)
            ck_threshold = 0.05
            ck_pass = ck_max_error < ck_threshold

            logger.info(
                "[msm] Chapman-Kolmogorov test completed: max_error=%.4f, pass=%s (threshold=%.4f)",
                ck_max_error,
                ck_pass,
                ck_threshold,
            )

            # Store CK metrics in MSM object
            msm.ck_max_error = ck_max_error
            msm.ck_pass = ck_pass
            msm.ck_threshold = ck_threshold
        else:
            logger.warning("[msm] No dtrajs available for CK test")
    except Exception as e:
        logger.warning("[msm] Chapman-Kolmogorov test failed: %s", e)
        # Keep infinite error and fail status on exception

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
                logger.info(
                    "[msm] 2D FES contour plot saved: fes_universal_ic1_vs_ic2.png"
                )
            else:
                logger.warning(
                    "[msm] No trajectories available for universal embedding"
                )
        except Exception as e:
            logger.error("[msm] Failed to generate universal embedding FES: %s", e)
    else:
        logger.debug(
            "[msm] Skipping phi/psi-specific FES (feature_type=%s)", feature_type
        )

    # Generate plots and analysis results with attribute checks
    logger.info("[msm] Generating analysis plots and results...")
    msm.plot_implied_timescales(save_file="implied_timescales")
    logger.debug("[msm] Implied timescales plot generated")
    msm.plot_free_energy_profile(save_file="free_energy_profile")
    logger.debug("[msm] Free energy profile plot generated")
    msm.create_state_table()
    logger.debug("[msm] State table created")
    msm.extract_representative_structures(save_pdb=True)
    logger.debug("[msm] Representative structures extracted")
    msm.save_analysis_results()
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
    logger.info(
        "[msm] Building MSM from labels: n_trajectories=%d, lag=%d", len(dtrajs), lag
    )
    if n_states is not None:
        logger.debug("[msm] Using provided n_states=%d", n_states)
    else:
        logger.debug("[msm] Inferring n_states from trajectory labels")

    T, pi = build_simple_msm(dtrajs, n_states=n_states, lag=lag)
    logger.info(
        "[msm] MSM built: transition_matrix_shape=%s, stationary_dist_shape=%s",
        T.shape,
        pi.shape,
    )
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
    logger.info(
        "[msm] Computing %d macrostates from transition matrix (shape=%s)",
        n_macrostates,
        T.shape,
    )
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
    logger.debug(
        "[msm] Computing macrostate populations: n_microstates=%d", len(pi_micro)
    )
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
    logger.debug(
        "[msm] Lumping micro to macro transition matrix: T_micro_shape=%s",
        T_micro.shape,
    )
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


def select_lag_with_ck_validation(
    shard_paths: List[Path | str],
    topology_path: Path | str,
    feature_spec_path: Path | str,
    n_clusters: int,
    tica_dim: int,
    tau_candidates: Optional[List[int]] = None,
    horizons: Optional[List[int]] = None,
    ck_threshold: float = 0.15,
    coverage_threshold: float = 0.98,
    min_median_count: int = 100,
    tica_lag: int = 10,
    diag_mass_threshold: float = 0.6,
) -> CKITSSelectionResult:
    """Select optimal lag using CK+ITS analysis on shard data.

    This function combines Chapman-Kolmogorov (CK) validation with Implied
    Timescales (ITS) analysis to automatically select the smallest lag time
    that passes validation criteria.

    The algorithm:
    1. Loads and aggregates shards
    2. Computes features and applies TICA reduction
    3. Clusters into microstates
    4. For each candidate lag:
       - Builds MSM at lag τ
       - Determines macrostates via PCCA+ (eigenvalue gap)
       - Predicts macro kinetics T^k
       - Observes macro kinetics at horizons kτ
       - Computes CK error
       - Checks sanity criteria (coverage, counts)
    5. Selects smallest lag passing all criteria

    Parameters
    ----------
    shard_paths : List[Path | str]
        List of shard JSON file paths.
    topology_path : Path | str
        Path to topology PDB file (for validation).
    feature_spec_path : Path | str
        Path to feature specification YAML file (for validation).
    n_clusters : int
        Number of microstates for clustering.
    tica_dim : int
        Number of TICA components to retain.
    tau_candidates : Optional[List[int]]
        Candidate lag times to evaluate. If None, uses [25, 50, 75, 100].
    horizons : Optional[List[int]]
        CK test horizons k. If None, uses [1, 2, 3, 4, 5].
    ck_threshold : float
        Maximum acceptable CK error (default: 0.15 = 15%).
    coverage_threshold : float
        Minimum coverage fraction (default: 0.98 = 98%).
    min_median_count : int
        Minimum median microstate count (default: 100).
    tica_lag : int
        Lag time for TICA dimensionality reduction (default: 10).
    diag_mass_threshold : float
        Minimum acceptable MSM diagonal mass (default: 0.6).

    Returns
    -------
    CKITSSelectionResult
        Result object containing:
        - selected_lag: The optimal lag time
        - ck_errors: CK error for each candidate lag
        - its_timescales: ITS data for plotting
        - coverage_fractions: Coverage for each lag
        - median_counts: Median counts for each lag
        - macrostate_counts: Number of macrostates for each lag
        - passed_sanity: Whether each lag passed sanity checks
        - diagnostics: Additional diagnostic information

    Raises
    ------
    FileNotFoundError
        If topology or feature spec files do not exist.
    ValueError
        If no shard files are provided or if data is insufficient.

    Examples
    --------
    >>> result = select_lag_with_ck_validation(
    ...     shard_paths=["shard1.json", "shard2.json"],
    ...     topology_path="protein.pdb",
    ...     feature_spec_path="features.yaml",
    ...     n_clusters=200,
    ...     tica_dim=10,
    ... )
    >>> print(f"Selected lag: {result.selected_lag}")
    """
    logger.info(
        "[CK-ITS API] Starting lag selection with %d shards, %d clusters, TICA dim %d",
        len(shard_paths),
        n_clusters,
        tica_dim,
    )

    # Validate inputs
    topo_path = Path(topology_path).expanduser().resolve()
    if not topo_path.exists():
        raise FileNotFoundError(f"Topology file not found: {topo_path}")

    spec_path = Path(feature_spec_path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Feature specification not found: {spec_path}")

    if not shard_paths:
        raise ValueError("No shard files provided")

    resolved_shards = [Path(p).expanduser().resolve() for p in shard_paths]

    # Load shards
    logger.info("[CK-ITS API] Loading %d shard files", len(resolved_shards))
    dataset = load_shards_as_dataset(resolved_shards)

    features = dataset.get("X")
    if features is None:
        raise ValueError("Aggregated shards did not contain feature matrix 'X'")

    X = np.asarray(features, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("Feature matrix must be 2D with at least one frame")

    n_frames, n_features = X.shape
    logger.info("[CK-ITS API] Loaded %d frames with %d features", n_frames, n_features)

    diag_payload = compute_diagnostics(
        {
            "__shards__": dataset.get("__shards__", []),
            "splits": {"aggregate": {"X": X}},
        }
    )
    autocorr_entry = diag_payload.get("autocorrelation", {}).get("aggregate", {}) or {}
    tau_int = autocorr_entry.get("tau_int")
    recommended_lags = autocorr_entry.get("recommended_ck_lags", [])
    lag_window = autocorr_entry.get("lag_window")
    derived_taus = diag_payload.get("taus", [])
    autocorr_curve = {
        "taus": autocorr_entry.get("taus"),
        "values": autocorr_entry.get("values"),
    }
    if not tau_candidates:
        tau_candidates = recommended_lags or derived_taus or [25, 50, 75, 100]
        logger.info(
            "[CK-ITS API] Tau candidates derived from autocorrelation: %s",
            tau_candidates,
        )

    # Apply TICA reduction
    from pmarlo.markov_state_model.reduction import reduce_features

    tica_components = min(tica_dim, X.shape[1])
    logger.info(
        "[CK-ITS API] Applying TICA reduction (lag=%d, components=%d)",
        tica_lag,
        tica_components,
    )
    Y = reduce_features(
        X,
        method="tica",
        lag=tica_lag,
        n_components=tica_components,
    )

    # Cluster into microstates
    from pmarlo.markov_state_model.clustering import cluster_microstates

    logger.info("[CK-ITS API] Clustering into %d microstates", n_clusters)
    clustering = cluster_microstates(
        Y,
        method="kmeans",
        n_states=n_clusters,
        random_state=42,
    )

    discrete = np.asarray(clustering.labels, dtype=np.int32)
    if discrete.size == 0:
        raise ValueError("Clustering produced no discrete states")

    unique_states = np.unique(discrete)
    logger.info(
        "[CK-ITS API] Clustered into %d unique states (requested %d)",
        unique_states.size,
        n_clusters,
    )

    # Prepare discrete trajectories
    dtrajs = [discrete]
    max_supported_lag = (
        max((int(traj.size) for traj in dtrajs if traj.size > 0), default=0) - 1
    )
    max_supported_lag = max(0, max_supported_lag)
    requested_tau_candidates = list(tau_candidates)

    # Run CK+ITS selection
    selected_lag, evaluations = select_optimal_lag_ck_its(
        dtrajs=dtrajs,
        tau_candidates=tau_candidates,
        horizons=horizons,
        ck_threshold=ck_threshold,
        coverage_threshold=coverage_threshold,
        min_median_count=min_median_count,
        diag_mass_threshold=diag_mass_threshold,
    )

    evaluated_lags = sorted({e.lag for e in evaluations})
    evaluated_set = set(evaluated_lags)
    ignored_tau_candidates = [
        tau for tau in requested_tau_candidates if tau not in evaluated_set
    ]

    # Collect ITS timescales from evaluations
    its_lags = []
    its_times = []
    for eval_result in evaluations:
        if eval_result.timescales is not None and eval_result.timescales.size > 0:
            its_lags.append(eval_result.lag)
            its_times.append(eval_result.timescales)

    # Pad timescales to same length for array conversion
    if its_times:
        max_len = max(ts.size for ts in its_times)
        padded_times = []
        for ts in its_times:
            if ts.size < max_len:
                padded = np.full(max_len, np.nan, dtype=float)
                padded[: ts.size] = ts
                padded_times.append(padded)
            else:
                padded_times.append(ts)
        its_timescales = np.array(padded_times)
    else:
        its_timescales = np.empty((0, 0), dtype=float)

    # Build result
    result = CKITSSelectionResult(
        selected_lag=selected_lag,
        ck_errors={e.lag: e.ck_error for e in evaluations},
        its_timescales=its_timescales,
        its_lag_times=np.array(its_lags, dtype=int),
        coverage_fractions={e.lag: e.coverage_fraction for e in evaluations},
        median_counts={e.lag: e.median_count for e in evaluations},
        macrostate_counts={e.lag: e.n_macrostates for e in evaluations},
        passed_sanity={e.lag: e.passed_sanity for e in evaluations},
        diagnostics={
            "n_frames": n_frames,
            "n_features": n_features,
            "n_clusters": n_clusters,
            "tica_dim": tica_components,
            "tica_lag": tica_lag,
            "n_unique_states": int(unique_states.size),
            "tau_int": tau_int,
            "recommended_tau_candidates": recommended_lags,
            "lag_window": lag_window,
            "autocorr_curve": autocorr_curve,
            "derived_tau_grid": derived_taus,
            "tau_candidates": requested_tau_candidates,
            "evaluated_tau_candidates": evaluated_lags,
            "ignored_tau_candidates": ignored_tau_candidates,
            "max_supported_lag": max_supported_lag,
            "ck_threshold": ck_threshold,
            "coverage_threshold": coverage_threshold,
            "min_median_count": min_median_count,
            "diag_mass_threshold": diag_mass_threshold,
            "evaluations": [
                {
                    "lag": e.lag,
                    "ck_error": e.ck_error,
                    "coverage": e.coverage_fraction,
                    "median_count": e.median_count,
                    "n_macrostates": e.n_macrostates,
                    "passed_sanity": e.passed_sanity,
                    "failure_reason": e.failure_reason,
                    "eigenvalue_gap": e.eigenvalue_gap,
                    "diag_mass": getattr(e, "diag_mass", None),
                }
                for e in evaluations
            ],
        },
    )

    logger.info("[CK-ITS API] Selection complete: selected_lag=%d", result.selected_lag)

    return result
