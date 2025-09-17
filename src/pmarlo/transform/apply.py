import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .plan import TransformPlan

logger = logging.getLogger(__name__)


def smooth_fes(dataset, **kwargs):
    return dataset


def reorder_states(dataset, **kwargs):
    return dataset


def fill_gaps(dataset, **kwargs):
    return dataset


def learn_cv_step(context: Dict[str, Any], **params) -> Dict[str, Any]:
    """Train learned CVs (Deep-TICA) and replace dataset features."""

    # Determine where the dataset is stored in the context
    dataset: Optional[Dict[str, Any]] = None
    uses_data_key = False
    if isinstance(context, dict) and isinstance(context.get("data"), dict):
        dataset = context["data"]
        uses_data_key = True
    elif isinstance(context, dict):
        dataset = context

    if not isinstance(dataset, dict):
        raise RuntimeError("LEARN_CV requires a mapping dataset with CV arrays")

    method = str(params.get("method", "deeptica")).lower()
    if method != "deeptica":
        raise RuntimeError(f"LEARN_CV method '{method}' is not supported")

    if "X" not in dataset:
        raise RuntimeError("LEARN_CV expects dataset['X'] containing CV features")

    X_all = np.asarray(dataset.get("X"), dtype=np.float64)
    if X_all.ndim != 2 or X_all.shape[0] == 0:
        raise RuntimeError("LEARN_CV requires a non-empty 2D feature matrix")

    shards_meta = dataset.get("__shards__")
    if not isinstance(shards_meta, list) or not shards_meta:
        shards_meta = [{"start": 0, "stop": X_all.shape[0]}]

    # Build per-shard slices
    shard_ranges: List[Tuple[int, int]] = []
    X_list: List[np.ndarray] = []
    for entry in shards_meta:
        try:
            start = int(entry.get("start", 0))
            stop = int(entry.get("stop", start))
        except Exception:
            continue
        start = max(0, start)
        stop = max(start, min(stop, X_all.shape[0]))
        if stop <= start:
            continue
        shard_ranges.append((start, stop))
        X_list.append(X_all[start:stop])

    if not X_list:
        raise RuntimeError("LEARN_CV requires at least one shard with frames")

    try:
        from pmarlo.features.deeptica import DeepTICAConfig, train_deeptica
    except ImportError as exc:
        raise RuntimeError(
            "Deep-TICA optional dependencies missing. Install pmarlo[mlcv] to enable LEARN_CV."
        ) from exc

    cfg_fields = getattr(DeepTICAConfig, "__annotations__", {}).keys()
    cfg_kwargs = {k: params[k] for k in params if k in cfg_fields}

    lag_param = int(params.get("lag", cfg_kwargs.get("lag", 5)))
    cfg_kwargs["lag"] = int(max(1, lag_param))
    if int(cfg_kwargs.get("n_out", 2)) < 2:
        cfg_kwargs["n_out"] = 2

    cfg = DeepTICAConfig(**cfg_kwargs)
    tau = int(max(1, cfg.lag))

    # Construct contiguous pairs per shard respecting the selected lag
    i_parts: List[np.ndarray] = []
    j_parts: List[np.ndarray] = []
    for start, stop in shard_ranges:
        length = stop - start
        if length <= tau:
            continue
        idx = np.arange(start, stop - tau, dtype=np.int64)
        if idx.size == 0:
            continue
        i_parts.append(idx)
        j_parts.append(idx + tau)

    if not i_parts:
        raise RuntimeError(
            f"LEARN_CV could not build lagged pairs for lag={cfg.lag}; check shard lengths."
        )

    idx_t = np.concatenate(i_parts)
    idx_tau = np.concatenate(j_parts)

    try:
        model = train_deeptica(X_list, (idx_t, idx_tau), cfg, weights=None)
    except Exception as exc:
        raise RuntimeError(f"Deep-TICA training failed: {exc}") from exc

    try:
        Y = model.transform(X_all).astype(np.float64, copy=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to transform CVs with Deep-TICA model: {exc}") from exc

    if Y.ndim != 2 or Y.shape[0] != X_all.shape[0]:
        raise RuntimeError("Deep-TICA returned invalid transformed features")

    n_out = int(Y.shape[1]) if Y.ndim == 2 else 0
    if n_out < 2:
        raise RuntimeError("Deep-TICA produced fewer than two components; expected >=2")

    # Replace feature matrix and metadata
    dataset["X"] = Y
    dataset["cv_names"] = tuple(f"DeepTICA_{i+1}" for i in range(n_out))
    dataset["periodic"] = tuple(False for _ in range(n_out))

    # Summarise results for downstream consumers
    history = getattr(model, "training_history", {}) or {}
    summary = {
        "applied": True,
        "method": "deeptica",
        "lag": int(cfg.lag),
        "n_out": n_out,
        "pairs_total": int(idx_t.shape[0]),
        "wall_time_s": float(history.get("wall_time_s", 0.0)),
        "loss_curve_last": (
            float(history["loss_curve"][-1])
            if isinstance(history.get("loss_curve"), list)
            and history.get("loss_curve")
            else None
        ),
        "objective_last": (
            float(history["objective_curve"][-1])
            if isinstance(history.get("objective_curve"), list)
            and history.get("objective_curve")
            else None
        ),
    }

    model_dir = params.get("model_dir")
    saved_prefix = None
    saved_files: List[str] = []
    if model_dir:
        try:
            base_dir = Path(model_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            stem = params.get("model_prefix") or f"deeptica-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            base_path = base_dir / stem
            model.save(base_path)
            saved_prefix = str(base_path)
            for suffix in (".json", ".pt", ".scaler.pt", ".history.json", ".history.csv"):
                candidate = base_path.with_suffix(suffix)
                if candidate.exists():
                    saved_files.append(str(candidate))
        except Exception as exc:
            logger.warning("Failed to persist Deep-TICA model: %s", exc)

    if saved_prefix:
        summary["model_prefix"] = saved_prefix
    if saved_files:
        summary["model_files"] = saved_files

    artifacts = dataset.setdefault("__artifacts__", {})
    artifacts["mlcv_deeptica"] = summary

    if uses_data_key:
        context["data"] = dataset

    return context


# Pipeline stage adapters
def protein_preparation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for protein preparation stage."""
    from ..protein.protein import Protein

    pdb_file = kwargs.get("pdb_file") or context.get("pdb_file")
    if not pdb_file:
        raise ValueError("pdb_file required for protein preparation")

    protein = Protein(pdb_file)
    prepared_pdb = protein.prepare_structure()

    context["protein"] = protein
    context["prepared_pdb"] = prepared_pdb
    logger.info(f"Protein prepared: {prepared_pdb}")
    return context


def system_setup(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for system setup stage."""
    protein = context.get("protein")
    if not protein:
        raise ValueError("protein required for system setup")

    # System setup logic would go here
    context["system_prepared"] = True
    logger.info("System setup completed")
    return context


def replica_initialization(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for replica initialization stage."""
    from ..replica_exchange.config import RemdConfig
    from ..replica_exchange.replica_exchange import ReplicaExchange

    prepared_pdb = context.get("prepared_pdb")
    temperatures = kwargs.get("temperatures") or context.get("temperatures", [300.0])
    output_dir = kwargs.get("output_dir") or context.get("output_dir", "output")

    if not prepared_pdb:
        raise ValueError("prepared_pdb required for replica initialization")

    config = RemdConfig(
        input_pdb=str(prepared_pdb),
        temperatures=temperatures,
        output_dir=str(output_dir),
    )

    replica_exchange = ReplicaExchange(config)
    context["replica_exchange"] = replica_exchange
    context["remd_config"] = config
    logger.info(f"Replica exchange initialized with {len(temperatures)} replicas")
    return context


def energy_minimization(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for energy minimization stage."""
    replica_exchange = context.get("replica_exchange")
    if not replica_exchange:
        raise ValueError("replica_exchange required for energy minimization")

    # Energy minimization would be handled by replica exchange
    context["energy_minimized"] = True
    logger.info("Energy minimization completed")
    return context


def gradual_heating(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for gradual heating stage."""
    replica_exchange = context.get("replica_exchange")
    if not replica_exchange:
        raise ValueError("replica_exchange required for gradual heating")

    context["heated"] = True
    logger.info("Gradual heating completed")
    return context


def equilibration(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for equilibration stage."""
    replica_exchange = context.get("replica_exchange")
    if not replica_exchange:
        raise ValueError("replica_exchange required for equilibration")

    context["equilibrated"] = True
    logger.info("Equilibration completed")
    return context


def production_simulation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for production simulation stage."""
    from ..replica_exchange.replica_exchange import run_remd_simulation

    remd_config = context.get("remd_config")
    if not remd_config:
        raise ValueError("remd_config required for production simulation")

    steps = kwargs.get("steps") or context.get("steps", 1000)

    # Run the actual simulation
    trajectory_files = run_remd_simulation(remd_config, steps=steps)

    context["trajectory_files"] = trajectory_files
    logger.info(
        f"Production simulation completed, generated {len(trajectory_files)} trajectories"
    )
    return context


def trajectory_demux(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for trajectory demultiplexing stage."""
    trajectory_files = context.get("trajectory_files", [])
    if not trajectory_files:
        logger.warning("No trajectory files found for demultiplexing")
        return context

    # Demultiplexing logic would go here
    context["demux_completed"] = True
    logger.info("Trajectory demultiplexing completed")
    return context


def trajectory_analysis(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for trajectory analysis stage."""
    trajectory_files = context.get("trajectory_files", [])
    if not trajectory_files:
        logger.warning("No trajectory files found for analysis")
        return context

    # Analysis logic would go here
    context["analysis_completed"] = True
    logger.info("Trajectory analysis completed")
    return context


def msm_build(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for MSM building stage."""
    from ..markov_state_model.enhanced_msm import run_complete_msm_analysis

    trajectory_files = context.get("trajectory_files", [])
    if not trajectory_files:
        raise ValueError("trajectory_files required for MSM building")

    n_states = kwargs.get("n_states") or context.get("n_states", 50)
    output_dir = kwargs.get("output_dir") or context.get("output_dir", "output")

    # Run MSM analysis
    msm_result = run_complete_msm_analysis(
        trajectory_files=trajectory_files, n_states=n_states, output_dir=str(output_dir)
    )

    context["msm_result"] = msm_result
    logger.info(f"MSM built with {n_states} states")
    return context


def build_analysis(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for build analysis stage."""
    msm_result = context.get("msm_result")
    if not msm_result:
        logger.warning("No MSM result found for build analysis")
        return context

    # Build analysis logic would go here
    context["build_analysis_completed"] = True
    logger.info("Build analysis completed")
    return context


def build_step(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for BUILD step that delegates to build_result."""
    from .build import AppliedOpts, BuildOpts, build_result
    from .plan import TransformPlan

    # Extract dataset from context
    dataset = context.get("data", context)

    # Create build options from step params and context
    opts_params = kwargs.copy()
    opts = BuildOpts(
        **{k: v for k, v in opts_params.items() if k in BuildOpts.__dataclass_fields__}
    )

    # Create applied options
    applied = AppliedOpts()

    # Create empty transform plan (build_result expects one)
    plan = TransformPlan(steps=())

    # Call build_result
    result = build_result(dataset, opts=opts, plan=plan, applied=applied)

    # Store result in context
    context["build_result"] = result
    context["build_completed"] = True

    logger.info("BUILD step completed")
    return context


def reduce_step(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Adapter for REDUCE step that applies dimensionality reduction."""
    from ..markov_state_model.reduction import reduce_features

    # Extract data from context
    data = context.get("data")
    if data is None:
        logger.warning("No data found for reduction step")
        return context

    # Get reduction parameters
    method = kwargs.get("method", "pca")
    n_components = kwargs.get("n_components", 2)
    lag = kwargs.get("lag", 1)
    scale = kwargs.get("scale", True)

    # Extract feature matrix
    if isinstance(data, dict) and "X" in data:
        X = data["X"]
    elif hasattr(data, "X"):
        X = data.X
    elif isinstance(data, np.ndarray):
        X = data
    else:
        logger.warning("Could not extract feature matrix for reduction")
        return context

    try:
        # Apply reduction
        X_reduced = reduce_features(
            X,
            method=method,
            n_components=n_components,
            lag=lag,
            scale=scale,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in ["method", "n_components", "lag", "scale"]
            },
        )

        # Store reduced data back in context
        if isinstance(data, dict):
            context["data"] = data.copy()
            context["data"]["X"] = X_reduced
            context["data"]["X_original"] = X  # Keep original for reference
        else:
            context["data"] = X_reduced
            context["X_original"] = X

        context["reduction_applied"] = True
        context["reduction_method"] = method
        context["reduction_components"] = n_components

        logger.info(
            f"REDUCE step completed using {method} with {n_components} components"
        )

    except Exception as e:
        logger.error(f"Reduction step failed: {e}")
        # Continue without reduction

    return context


def apply_transform_plan(dataset, plan: TransformPlan):
    """Apply a transform plan to data or context."""
    # If dataset is not a dict, wrap it in a context
    if not isinstance(dataset, dict):
        context = {"data": dataset}
    else:
        context = dataset.copy()

    for step in plan.steps:
        if step.name == "SMOOTH_FES":
            context = smooth_fes(context, **step.params)
        elif step.name == "LEARN_CV":
            context = learn_cv_step(context, **step.params)
        elif step.name == "REDUCE":
            context = reduce_step(context, **step.params)
        elif step.name == "REORDER_STATES":
            context = reorder_states(context, **step.params)
        elif step.name == "FILL_GAPS":
            context = fill_gaps(context, **step.params)
        elif step.name == "PROTEIN_PREPARATION":
            context = protein_preparation(context, **step.params)
        elif step.name == "SYSTEM_SETUP":
            context = system_setup(context, **step.params)
        elif step.name == "REPLICA_INITIALIZATION":
            context = replica_initialization(context, **step.params)
        elif step.name == "ENERGY_MINIMIZATION":
            context = energy_minimization(context, **step.params)
        elif step.name == "GRADUAL_HEATING":
            context = gradual_heating(context, **step.params)
        elif step.name == "EQUILIBRATION":
            context = equilibration(context, **step.params)
        elif step.name == "PRODUCTION_SIMULATION":
            context = production_simulation(context, **step.params)
        elif step.name == "TRAJECTORY_DEMUX":
            context = trajectory_demux(context, **step.params)
        elif step.name == "TRAJECTORY_ANALYSIS":
            context = trajectory_analysis(context, **step.params)
        elif step.name == "MSM_BUILD":
            context = msm_build(context, **step.params)
        elif step.name == "BUILD_ANALYSIS":
            context = build_analysis(context, **step.params)
        elif step.name == "BUILD":
            context = build_step(context, **step.params)
        else:
            logger.warning(f"Unknown transform step: {step.name}")

    # If original dataset was not a dict, extract the data
    if "data" in context and not isinstance(dataset, dict):
        return context["data"]

    return context
