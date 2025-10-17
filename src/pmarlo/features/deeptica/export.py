"""Export trained DeepTICA models for OpenMM integration."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from pmarlo.utils.path_utils import ensure_directory

logger = logging.getLogger(__name__)

__all__ = ["CVModelBundle", "export_cv_model", "load_cv_model_info"]


@dataclass(slots=True)
class CVModelBundle:
    """Container for exported CV model files and metadata."""

    model_path: Path  # Path to TorchScript (.pt) file
    scaler_path: Path  # Path to scaler parameters (.npz)
    config_path: Path  # Path to model configuration (.json)
    metadata_path: Path  # Path to export metadata (.json)
    feature_names: list[str]  # Names of input features
    cv_dim: int  # Dimension of collective variable output


def export_cv_model(
    network: nn.Module,
    scaler: Any,
    history: dict[str, Any],
    output_dir: str | Path,
    *,
    model_name: str = "deeptica_cv_model",
    feature_names: Optional[list[str]] = None,
) -> CVModelBundle:
    """
    Export a trained DeepTICA model for use in OpenMM simulations.

    This function exports the model in TorchScript format compatible with
    openmm-torch, along with scaler parameters and metadata needed for
    integration into molecular dynamics simulations.

    Parameters
    ----------
    network : nn.Module
        Trained DeepTICA network (can be wrapped with preprocessing layers)
    scaler : Any
        Fitted StandardScaler or similar scaler used during training
    history : dict[str, Any]
        Training history containing configuration and metrics
    output_dir : str | Path
        Directory where exported files will be saved
    model_name : str, optional
        Base name for exported files (default: "deeptica_cv_model")
    feature_names : list[str], optional
        Names of input features (e.g., ["distance_1_2", "angle_3_4_5"])
        If None, generic names will be generated

    Returns
    -------
    CVModelBundle
        Bundle containing paths to all exported files and metadata

    Raises
    ------
    RuntimeError
        If model export fails (e.g., model contains unsupported operations)
    ValueError
        If model or scaler are invalid

    Notes
    -----
    The exported model can be loaded in OpenMM using openmm-torch:
    
    >>> import torch
    >>> from openmmtorch import TorchForce
    >>> model = torch.jit.load('deeptica_cv_model.pt')
    >>> cv_force = TorchForce(model)
    >>> system.addForce(cv_force)

    The scaler must be applied to input features before passing to the model:
    
    >>> X_scaled = (X - scaler_mean) / scaler_scale
    >>> cvs = model(X_scaled)
    """
    output_path = Path(output_dir)
    ensure_directory(output_path)

    # Validate inputs
    if network is None:
        raise ValueError("Network cannot be None")
    if scaler is None:
        raise ValueError("Scaler cannot be None")

    # Extract scaler parameters
    scaler_mean = np.asarray(getattr(scaler, "mean_", []), dtype=np.float32)
    scaler_scale = np.asarray(getattr(scaler, "scale_", []), dtype=np.float32)
    
    if scaler_mean.size == 0 or scaler_scale.size == 0:
        raise ValueError("Scaler has not been fitted (missing mean_ or scale_)")

    input_dim = int(scaler_mean.shape[0])
    
    # Infer output dimension from network
    network.eval()
    with torch.no_grad():
        dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)
        try:
            dummy_output = network(dummy_input)
            cv_dim = int(dummy_output.shape[-1])
        except Exception as exc:
            logger.warning(
                "Could not infer CV dimension from network forward pass: %s", exc
            )
            cv_dim = int(history.get("n_out", 2))

    # Generate feature names if not provided
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(input_dim)]
    
    if len(feature_names) != input_dim:
        logger.warning(
            "Feature names length (%d) does not match input dimension (%d), "
            "generating generic names",
            len(feature_names),
            input_dim,
        )
        feature_names = [f"feature_{i}" for i in range(input_dim)]

    # Export model as TorchScript
    model_path = output_path / f"{model_name}.pt"
    try:
        # Set model to eval mode and trace it
        network.eval()
        example_input = torch.zeros(1, input_dim, dtype=torch.float32)
        
        # Use torch.jit.trace for better compatibility with openmm-torch
        traced_model = torch.jit.trace(network, example_input)
        
        # Save the traced model
        torch.jit.save(traced_model, str(model_path))
        logger.info("Exported TorchScript model to %s", model_path)
        
    except Exception as exc:
        raise RuntimeError(
            f"Failed to export model as TorchScript: {exc}. "
            "Ensure the model does not contain dynamic control flow or "
            "unsupported operations."
        ) from exc

    # Export scaler parameters
    scaler_path = output_path / f"{model_name}_scaler.npz"
    np.savez(
        scaler_path,
        mean=scaler_mean,
        scale=scaler_scale,
        feature_names=np.array(feature_names, dtype=object),
    )
    logger.info("Exported scaler parameters to %s", scaler_path)

    # Export model configuration
    config_path = output_path / f"{model_name}_config.json"
    config_dict = {
        "input_dim": input_dim,
        "cv_dim": cv_dim,
        "feature_names": feature_names,
        "activation": history.get("activation", "gelu"),
        "hidden_layers": history.get("hidden", [32, 16]),
        "tau_schedule": history.get("tau_schedule", []),
        "vamp2_score": history.get("vamp2_after"),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    logger.info("Exported model configuration to %s", config_path)

    # Export metadata
    metadata_path = output_path / f"{model_name}_metadata.json"
    metadata = {
        "model_name": model_name,
        "export_format": "torchscript",
        "openmm_compatible": True,
        "requires_openmm_torch": True,
        "model_type": "deeptica",
        "input_dim": input_dim,
        "cv_dim": cv_dim,
        "training_history": {
            "vamp2_before": history.get("vamp2_before"),
            "vamp2_after": history.get("vamp2_after"),
            "epochs_per_tau": history.get("epochs_per_tau"),
            "tau_schedule": history.get("tau_schedule"),
            "wall_time_s": history.get("wall_time_s"),
        },
        "usage": {
            "load_model": f"torch.jit.load('{model_name}.pt')",
            "load_scaler": f"np.load('{model_name}_scaler.npz')",
            "openmm_integration": "Use openmmtorch.TorchForce to add as custom force",
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Exported model metadata to %s", metadata_path)

    return CVModelBundle(
        model_path=model_path,
        scaler_path=scaler_path,
        config_path=config_path,
        metadata_path=metadata_path,
        feature_names=feature_names,
        cv_dim=cv_dim,
    )


def load_cv_model_info(bundle_dir: str | Path, model_name: str = "deeptica_cv_model") -> dict[str, Any]:
    """
    Load metadata and configuration from an exported CV model bundle.

    Parameters
    ----------
    bundle_dir : str | Path
        Directory containing exported model files
    model_name : str, optional
        Base name of the exported model files

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - model_path: Path to TorchScript file
        - scaler_path: Path to scaler file
        - config: Model configuration dict
        - metadata: Export metadata dict
        - scaler_params: Loaded scaler parameters (mean, scale)

    Raises
    ------
    FileNotFoundError
        If required model files are not found
    """
    bundle_path = Path(bundle_dir)
    
    model_path = bundle_path / f"{model_name}.pt"
    scaler_path = bundle_path / f"{model_name}_scaler.npz"
    config_path = bundle_path / f"{model_name}_config.json"
    metadata_path = bundle_path / f"{model_name}_metadata.json"

    # Check required files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

    # Load configuration
    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Load metadata
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Load scaler parameters
    scaler_data = np.load(scaler_path, allow_pickle=True)
    scaler_params = {
        "mean": scaler_data["mean"],
        "scale": scaler_data["scale"],
        "feature_names": scaler_data.get("feature_names", []).tolist(),
    }

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "config": config,
        "metadata": metadata,
        "scaler_params": scaler_params,
    }

