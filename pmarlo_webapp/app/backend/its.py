
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pmarlo.data.aggregate import load_shards_as_dataset

def plot_its(
    lag_times: Sequence[int],
    timescale_series: Sequence[Sequence[float] | np.ndarray],
    *,
    max_timescales: int = 10,
):
    """Generate an implied-timescale convergence plot."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("matplotlib is required to plot implied timescales") from exc

    lags = [int(lag) for lag in lag_times]
    if not lags:
        raise ValueError("At least one lag time is required to plot ITS results")

    series = [np.asarray(ts, dtype=float).reshape(-1) for ts in timescale_series]
    if len(series) != len(lags):
        raise ValueError("Lag time list and timescale series length mismatch")

    max_len = max((arr.size for arr in series), default=0)
    if max_len == 0:
        raise ValueError("No implied timescales available to plot")

    n_curves = min(int(max_len), int(max_timescales))

    fig, ax = plt.subplots()
    for idx in range(n_curves):
        y_vals: List[float] = []
        for arr in series:
            if arr.size > idx and np.isfinite(arr[idx]) and arr[idx] > 0:
                y_vals.append(float(arr[idx]))
            else:
                y_vals.append(np.nan)
        ax.plot(lags, y_vals, marker="o", label=f"Timescale {idx + 1}")

    ax.set_xlabel("Lag time (steps)")
    ax.set_ylabel("Implied timescale (steps)")
    ax.set_title("Implied Timescales Convergence")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig

def calculate_its(
    data_directory: Path | str,
    topology_path: Path | str,
    feature_spec_path: Path | str,
    n_clusters: int,
    tica_dim: int,
    lag_times: Sequence[int | float],
    *,
    shard_paths: Optional[Sequence[Path | str]] = None,
    tica_lag: int = 10,
) -> Dict[str, Any]:
    """Compute implied timescales for a dataset prior to full MSM analysis.

    Parameters
    ----------
    data_directory:
        Directory containing shard JSON files. Used when ``shard_paths`` is
        not provided.
    topology_path:
        Path to the topology PDB file associated with the trajectories.
        The file must exist; the path is validated but otherwise unused here.
    feature_spec_path:
        Path to a ``feature_spec.yaml`` description. The file must exist and
        contain a valid YAML payload.
    n_clusters:
        Number of clusters used when discretising the reduced features.
    tica_dim:
        Number of TICA components to retain when projecting the feature data.
    lag_times:
        Iterable of lag times (in MD steps) to evaluate during the ITS scan.
    shard_paths:
        Optional explicit list of shard JSON paths. When omitted, all shard
        JSON files under ``data_directory`` are used.
    tica_lag:
        Lag time used for TICA dimensionality reduction.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing lag times, implied timescales, metadata, and
        per-lag error messages, if any.
    """

    if dt is None:  # pragma: no cover - exercised only when optional dep missing
        raise RuntimeError(
            "deeptime is required to compute implied timescales but is not installed."
        )

    data_root = Path(data_directory).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory {data_root} does not exist")

    if shard_paths is None:
        shard_candidates = sorted(data_root.rglob("*.json"))
    else:
        shard_candidates = [Path(p).expanduser().resolve() for p in shard_paths]

    if not shard_candidates:
        raise ValueError("No shard JSON files available for implied timescale analysis")

    topo_candidate = Path(topology_path).expanduser()
    if not topo_candidate.is_absolute():
        topo_candidate = (data_root / topo_candidate).resolve()
    else:
        topo_candidate = topo_candidate.resolve()
    if not topo_candidate.exists():
        raise FileNotFoundError(f"Topology file {topo_candidate} does not exist")

    spec_path = Path(feature_spec_path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Feature specification file {spec_path} does not exist")

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - PyYAML is part of dependencies
        raise RuntimeError("PyYAML is required to load feature specifications") from exc

    with spec_path.open("r", encoding="utf-8") as handle:
        spec_payload = yaml.safe_load(handle)
    if spec_payload is None:
        raise ValueError(f"Feature specification at {spec_path} is empty")

    aggregated = load_shards_as_dataset(shard_candidates)
    features = aggregated.get("X")
    if features is None:
        raise ValueError("Aggregated shards did not contain feature matrix 'X'")

    X = np.asarray(features, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("Feature matrix must be two-dimensional with at least one frame")

    n_frames, n_features = X.shape
    logger.info(
        "[ITS] Loaded %d frames with %d features from %d shard files",
        n_frames,
        n_features,
        len(shard_candidates),
    )

    if tica_dim < 1:
        raise ValueError("tica_dim must be a positive integer")
    if n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer")

    from pmarlo.markov_state_model.reduction import reduce_features
    from pmarlo.markov_state_model.clustering import cluster_microstates

    tica_components = min(int(tica_dim), int(X.shape[1]))
    logger.info(
        "[ITS] Applying TICA reduction (lag=%d, components=%d)",
        int(tica_lag),
        tica_components,
    )
    Y = reduce_features(
        X,
        method="tica",
        lag=int(tica_lag),
        n_components=tica_components,
    )

    clustering = cluster_microstates(
        Y,
        method="kmeans",
        n_states=int(n_clusters),
        random_state=42,
    )

    discrete = np.asarray(clustering.labels, dtype=np.int32)
    if discrete.size == 0:
        raise ValueError("Clustering produced no discrete states; cannot compute ITS")

    unique_states = np.unique(discrete)
    logger.info(
        "[ITS] Clustered trajectories into %d states (requested %d)",
        unique_states.size,
        int(n_clusters),
    )

    lags_sorted = sorted({int(max(1, int(lag))) for lag in lag_times})
    if not lags_sorted:
        raise ValueError("Provide at least one positive lag time for ITS analysis")

    from deeptime.markov.msm import MaximumLikelihoodMSM

    timescale_results: List[np.ndarray] = []
    errors: Dict[int, str] = {}

    dtrajs = [discrete]

    for lag in lags_sorted:
        logger.info("[ITS] Estimating MSM at lag=%d", lag)
        estimator = MaximumLikelihoodMSM(lagtime=int(lag), reversible=True)
        try:
            fit_result = estimator.fit(dtrajs)
            msm_model = fit_result.fetch_model()
            times = np.asarray(msm_model.timescales(), dtype=float)
        except Exception as exc:  # pragma: no cover - depends on runtime data
            errors[int(lag)] = str(exc)
            logger.error("[ITS] MSM estimation failed for lag=%d: %s", lag, exc)
            times = np.asarray([], dtype=float)
        timescale_results.append(times)

    metadata = {
        "n_frames": int(n_frames),
        "n_features": int(n_features),
        "tica_lag": int(tica_lag),
        "tica_dim": int(tica_components),
        "n_clusters": int(n_clusters),
        "n_states": int(unique_states.size),
        "topology_path": str(topo_candidate),
        "feature_spec_path": str(spec_path),
        "n_shards": len(shard_candidates),
    }

    return {
        "lag_times": lags_sorted,
        "timescales": timescale_results,
        "errors": errors,
        "metadata": metadata,
    }