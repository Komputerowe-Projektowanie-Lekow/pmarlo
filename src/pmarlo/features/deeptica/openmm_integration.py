"""OpenMM integration for Deep-TICA collective variable models."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

_PLUGIN_PATTERNS = (
    "OpenMMTorch*.dll",
    "*Torch*.dll",
    "libOpenMMTorch*.so",
    "libOpenMMTorch*.dylib",
)

_PLUGIN_SUBDIRS = (
    Path("Library") / "plugins",
    Path("lib") / "plugins",
    Path("lib64") / "plugins",
)

__all__ = [
    "CVBiasForce",
    "add_cv_bias_to_system",
    "create_cv_torch_force",
    "check_openmm_torch_available",
]


def _path_from_string(value: str | None) -> Optional[Path]:
    if not value:
        return None
    try:
        return Path(value).expanduser()
    except Exception:
        return None


def _split_paths(raw: str | None) -> Iterable[Path]:
    if not raw:
        return []
    return filter(None, (_path_from_string(part) for part in raw.split(os.pathsep)))


def _contains_torch_plugin(directory: Path) -> bool:
    if not directory or not directory.is_dir():
        return False
    for pattern in _PLUGIN_PATTERNS:
        if next(directory.glob(pattern), None):
            return True
    return False


def _collect_env_plugin_dirs() -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def register(path: Path | None, *, require_plugin: bool = False) -> None:
        if path is None:
            return
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            return
        if require_plugin and not _contains_torch_plugin(resolved):
            return
        if not resolved.exists():
            return
        seen.add(resolved)
        candidates.append(resolved)

    # Explicit env overrides
    for var in ("PMARLO_OPENMM_PLUGIN_DIR", "OPENMM_PLUGIN_DIR"):
        register(_path_from_string(os.environ.get(var)))
    for var in ("PMARLO_OPENMM_PLUGIN_DIRS", "OPENMM_PLUGIN_DIRS"):
        for path in _split_paths(os.environ.get(var)):
            register(path)

    def add_env_prefix(prefix: Path | None) -> None:
        if prefix is None:
            return
        register(prefix / "Library" / "plugins")
        register(prefix / "lib" / "plugins")
        register(prefix / "lib64" / "plugins")

        parent = prefix.parent
        envs_root = parent if parent.name == "envs" else parent / "envs"
        if envs_root.exists():
            for env_dir in envs_root.iterdir():
                if not env_dir.is_dir():
                    continue
                for sub in _PLUGIN_SUBDIRS:
                    register(env_dir / sub, require_plugin=True)

    add_env_prefix(_path_from_string(os.environ.get("CONDA_PREFIX")))

    for var in ("CONDA_ENVS_PATH", "MAMBA_ROOT_PREFIX", "MICROMAMBA_ROOT_PREFIX"):
        for root in _split_paths(os.environ.get(var)):
            envs_root = root / "envs" if (root / "envs").exists() else root
            if envs_root.exists():
                for env_dir in envs_root.iterdir():
                    if not env_dir.is_dir():
                        continue
                    for sub in _PLUGIN_SUBDIRS:
                        register(env_dir / sub, require_plugin=True)

    # Current interpreter prefix
    add_env_prefix(Path(sys.prefix))
    return candidates


def _resolve_torchforce_class():
    try:
        from openmmtorch import TorchForce  # type: ignore

        return TorchForce
    except ImportError as first_exc:
        try:
            import openmm
            from openmm import Platform
        except Exception as exc:
            raise ImportError("OpenMM is not available") from exc

        plugin_dirs: list[Path] = []
        try:
            default_dir = Platform.getDefaultPluginsDirectory()
            if default_dir:
                plugin_dirs.append(Path(default_dir))
        except Exception:
            default_dir = None

        plugin_dirs.extend(_collect_env_plugin_dirs())

        loaded_any = False
        for directory in plugin_dirs:
            try:
                if directory and directory.exists():
                    Platform.loadPluginsFromDirectory(str(directory))
                    loaded_any = True
            except Exception as exc:
                logger.debug(
                    "Failed to load OpenMM plugins from %s: %s", directory, exc
                )

        torch_force = getattr(openmm, "TorchForce", None)
        if torch_force is None:
            torch_force = getattr(
                getattr(openmm, "openmm", object()), "TorchForce", None
            )

        if torch_force is None:
            hint = (
                "TorchForce class not found in OpenMM plugins. "
                "Install openmm-torch in the same environment or set "
                "PMARLO_OPENMM_PLUGIN_DIR/OPENMM_PLUGIN_DIR to the plugin directory."
            )
            if not loaded_any:
                hint += " (No plugin directories could be loaded automatically.)"
            raise ImportError(hint) from first_exc

        return torch_force


def check_openmm_torch_available() -> bool:
    """
    Check if openmm-torch is available for CV model integration.

    Returns
    -------
    bool
        True if the TorchForce class can be resolved, False otherwise.
    """
    try:
        _resolve_torchforce_class()
        return True
    except Exception as exc:  # pragma: no cover - diagnostic path
        logger.debug("TorchForce unavailable: %s", exc)
        return False


def create_cv_torch_force(
    model_path: str | Path,
    *,
    force_group: int = 0,
    use_cuda: bool = True,
) -> Any:
    """
    Create an OpenMM TorchForce from a trained CV model.

    This function loads a TorchScript model and wraps it in an OpenMM TorchForce
    that can be added to an OpenMM System. The model will compute collective
    variables during the simulation and can be used for biased sampling.

    Parameters
    ----------
    model_path : str | Path
        Path to the TorchScript model file (.pt)
    force_group : int, optional
        OpenMM force group for the CV force (default: 0)
    use_cuda : bool, optional
        Whether to use CUDA for model evaluation (default: True)

    Returns
    -------
    TorchForce
        OpenMM TorchForce object containing the CV model

    Raises
    ------
    ImportError
        If openmm-torch is not installed
    FileNotFoundError
        If model file does not exist
    RuntimeError
        If model loading fails

    Notes
    -----
    Requires openmm-torch to be installed:

        conda install -c conda-forge openmm-torch

    The TorchForce expects input features to already be scaled using the
    scaler parameters. Feature extraction and scaling should be handled
    by OpenMM CustomCVForce or similar mechanisms before the model input.

    Example
    -------
    >>> cv_force = create_cv_torch_force("deeptica_cv_model.pt")
    >>> system.addForce(cv_force)
    """
    if not check_openmm_torch_available():
        raise ImportError(
            "openmm-torch is required for CV model integration. "
            "Install it with: conda install -c conda-forge openmm-torch"
        )

    import torch

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"CV model file not found: {model_path}")

    try:
        # Load the TorchScript model
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        logger.info("Loaded CV model from %s on device %s", model_path, device)

    except Exception as exc:
        raise RuntimeError(f"Failed to load CV model from {model_path}: {exc}") from exc

    # Create TorchForce
    TorchForce = _resolve_torchforce_class()
    try:
        torch_force = TorchForce(str(model_path))
        torch_force.setForceGroup(force_group)
        logger.info("Created TorchForce with force group %d", force_group)
        return torch_force
    except Exception as exc:
        raise RuntimeError(f"Failed to create TorchForce: {exc}") from exc


class CVBiasForce:
    """
    Wrapper for CV-based biasing forces in OpenMM simulations.

    This class manages the integration of a trained Deep-TICA model as a
    biasing force in OpenMM. It handles model loading, feature extraction,
    and force application.

    Parameters
    ----------
    model_path : str | Path
        Path to the TorchScript CV model file
    scaler_mean : np.ndarray
        Mean values from training scaler
    scaler_scale : np.ndarray
        Scale values from training scaler
    bias_strength : float, optional
        Strength of the biasing force (default: 1.0)
    use_cuda : bool, optional
        Whether to use CUDA for model evaluation (default: True)

    Attributes
    ----------
    model_path : Path
        Path to the model file
    scaler_mean : np.ndarray
        Scaler mean parameters
    scaler_scale : np.ndarray
        Scaler scale parameters
    bias_strength : float
        Strength of the biasing force
    force : TorchForce or None
        The OpenMM TorchForce object (created when added to system)
    """

    def __init__(
        self,
        model_path: str | Path,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        *,
        bias_strength: float = 1.0,
        use_cuda: bool = True,
    ):
        self.model_path = Path(model_path)
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        self.bias_strength = float(bias_strength)
        self.use_cuda = bool(use_cuda)
        self.force: Optional[Any] = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"CV model file not found: {self.model_path}")

    def add_to_system(self, system: Any, force_group: int = 0) -> None:
        """
        Add the CV bias force to an OpenMM System.

        Parameters
        ----------
        system : openmm.System
            The OpenMM System to add the force to
        force_group : int, optional
            OpenMM force group for the CV force (default: 0)

        Raises
        ------
        ImportError
            If openmm-torch is not installed
        RuntimeError
            If force creation fails
        """
        self.force = create_cv_torch_force(
            self.model_path,
            force_group=force_group,
            use_cuda=self.use_cuda,
        )
        system.addForce(self.force)
        logger.info(
            "Added CV bias force to system (strength=%.3f, group=%d)",
            self.bias_strength,
            force_group,
        )

    def get_cv_values(self, simulation: Any) -> Optional[np.ndarray]:
        """
        Get current CV values from the simulation.

        Parameters
        ----------
        simulation : openmm.app.Simulation
            The OpenMM Simulation object

        Returns
        -------
        np.ndarray or None
            Current CV values, or None if not available

        Notes
        -----
        This method extracts the current collective variable values by
        querying the state of the simulation. This can be used for
        monitoring or adaptive sampling strategies.
        """
        if self.force is None:
            logger.warning("CV force has not been added to a system yet")
            return None

        try:
            # Get state with forces
            simulation.context.getState(
                getEnergy=True, groups={self.force.getForceGroup()}
            )
            # CV values would need to be extracted from the force
            # This is a placeholder - actual implementation depends on openmm-torch API
            logger.debug("CV values extraction not yet implemented")
            return None

        except Exception as exc:
            logger.warning("Failed to extract CV values: %s", exc)
            return None


def add_cv_bias_to_system(
    system: Any,
    model_path: str | Path,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    *,
    bias_strength: float = 1.0,
    force_group: int = 0,
    use_cuda: bool = True,
) -> CVBiasForce:
    """
    Convenience function to add CV bias force to an OpenMM system.

    Parameters
    ----------
    system : openmm.System
        The OpenMM System to add the force to
    model_path : str | Path
        Path to the TorchScript CV model file
    scaler_mean : np.ndarray
        Mean values from training scaler
    scaler_scale : np.ndarray
        Scale values from training scaler
    bias_strength : float, optional
        Strength of the biasing force (default: 1.0)
    force_group : int, optional
        OpenMM force group for the CV force (default: 0)
    use_cuda : bool, optional
        Whether to use CUDA for model evaluation (default: True)

    Returns
    -------
    CVBiasForce
        The created and added CV bias force

    Raises
    ------
    ImportError
        If openmm-torch is not installed
    FileNotFoundError
        If model file does not exist
    RuntimeError
        If force creation or addition fails

    Example
    -------
    >>> # Load CV model info
    >>> info = load_cv_model_info("path/to/model/dir")
    >>> # Add to OpenMM system
    >>> cv_force = add_cv_bias_to_system(
    ...     system,
    ...     info["model_path"],
    ...     info["scaler_params"]["mean"],
    ...     info["scaler_params"]["scale"],
    ...     bias_strength=0.5,
    ... )
    """
    cv_bias = CVBiasForce(
        model_path=model_path,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        bias_strength=bias_strength,
        use_cuda=use_cuda,
    )
    cv_bias.add_to_system(system, force_group=force_group)
    return cv_bias
