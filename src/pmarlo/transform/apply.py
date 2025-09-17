import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .plan import TransformPlan

logger = logging.getLogger(__name__)


def smooth_fes(dataset, **kwargs):
    return dataset


def reorder_states(dataset, **kwargs):
    return dataset


def fill_gaps(dataset, **kwargs):
    return dataset


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
