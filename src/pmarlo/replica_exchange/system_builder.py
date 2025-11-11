from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import openmm
import torch
from openmm import unit
from openmm.app import PME, ForceField, HBonds, PDBFile

from pmarlo import constants as const
from pmarlo.features.deeptica import check_openmm_torch_available
from pmarlo.features.deeptica.export import load_cv_model_info
from pmarlo.settings import ensure_scaler_finite, load_defaults, load_feature_spec

logger = logging.getLogger(__name__)


def load_pdb_and_forcefield(
    pdb_file: str, forcefield_files: List[str]
) -> Tuple[PDBFile, ForceField]:
    pdb = PDBFile(pdb_file)
    forcefield = ForceField(*forcefield_files)
    return pdb, forcefield


def _normalise_path_string(value: str | Path | None) -> str:
    """Normalise filesystem paths so metadata comparisons work across OSes."""
    if value is None:
        return ""
    text = str(value).strip().strip('"')
    if not text:
        return ""
    candidate = text.replace("\\", "/")
    if candidate.startswith("/mnt/") and len(candidate) > 6:
        drive_letter = candidate[5]
        remainder = candidate[7:]
        if candidate[6] == "/" and drive_letter.isalpha():
            candidate = f"{drive_letter.upper()}:/{remainder}"
    try:
        path = Path(candidate)
    except Exception:
        return text
    path = path.expanduser()
    try:
        resolved = path.resolve(strict=False)
    except Exception:
        resolved = path
    return str(resolved)


def _collect_recorded_model_paths(payload: Mapping[str, Any] | None) -> list[str]:
    """Pull every recorded model path in metadata/history for bundle discovery."""
    if not isinstance(payload, Mapping):
        return []
    values: list[str] = []
    for key in ("model_prefix", "model_path"):
        raw = payload.get(key)
        if isinstance(raw, str):
            values.append(raw)
    files = payload.get("model_files")
    if isinstance(files, list):
        for item in files:
            if isinstance(item, str):
                values.append(item)
    return values


def _find_existing_cv_bundle_dir(checkpoint_path: Path) -> Path | None:
    """Locate the exported CV bundle directory associated with a checkpoint."""
    base_prefix = checkpoint_path.with_suffix("")
    try:
        base_prefix = base_prefix.expanduser().resolve(strict=False)
    except Exception:
        base_prefix = base_prefix.expanduser()

    base_norm = _normalise_path_string(base_prefix)
    base_name = base_prefix.name

    parent = base_prefix.parent
    search_dirs: list[Path] = []
    if parent.is_dir():
        search_dirs.append(parent)
    for candidate in sorted(parent.glob("training-*")):
        if candidate.is_dir():
            search_dirs.append(candidate)

    for directory in search_dirs:
        ts_path = directory / "deeptica_cv_model.pt"
        if not ts_path.exists():
            continue
        meta_path = directory / "deeptica_cv_model_metadata.json"
        recorded_match = False
        recorded_names: set[str] = set()

        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
            history = metadata.get("history")
            recorded_values = _collect_recorded_model_paths(metadata) + _collect_recorded_model_paths(
                history
            )
            for raw in recorded_values:
                normalised = _normalise_path_string(raw)
                if normalised and base_norm and normalised == base_norm:
                    recorded_match = True
                    break
                try:
                    recorded_names.add(Path(raw).with_suffix("").name)
                except Exception:
                    continue
            if not recorded_match and base_name in recorded_names:
                recorded_match = True

        if recorded_match:
            return directory

    return None


@dataclass(frozen=True)
class _BiasConfig:
    enabled: bool
    mode: str
    torch_threads: int
    precision: str


def _build_base_system(pdb: PDBFile, forcefield: ForceField) -> openmm.System:
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        constraints=HBonds,
        rigidWater=True,
        nonbondedCutoff=0.9 * unit.nanometer,
        ewaldErrorTolerance=const.REPLICA_EXCHANGE_EWALD_TOLERANCE,
        hydrogenMass=3.0 * unit.amu,
    )
    has_cmm = any(
        isinstance(system.getForce(i), openmm.CMMotionRemover)
        for i in range(system.getNumForces())
    )
    if not has_cmm:
        system.addForce(openmm.CMMotionRemover())
    return system


def _load_bias_config() -> _BiasConfig:
    config = load_defaults()
    return _BiasConfig(
        enabled=bool(config["enable_cv_bias"]),
        mode=str(config["bias_mode"]).lower(),
        torch_threads=int(config["torch_threads"]),
        precision=str(config["precision"]).lower(),
    )


def _ensure_bias_disabled(cv_model_path: str | None) -> None:
    if cv_model_path is not None:
        raise RuntimeError(
            "CV bias disabled via configuration but cv_model_path was provided. "
            "Set enable_cv_bias=true to activate the bias."
        )


def _get_jit_attribute(module: torch.jit.ScriptModule, name: str):
    """Safely read an attribute that may be stored on either the Python or compiled module."""
    try:
        return getattr(module, name)
    except AttributeError:
        pass
    try:
        return module._c.getattr(name)
    except Exception:
        return None


def _tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _extract_scaler_from_script_module(module: torch.jit.ScriptModule) -> dict[str, Any]:
    """Pull scaler statistics and feature names directly from a TorchScript bias module."""
    mean_tensor = _get_jit_attribute(module, "scaler_mean")
    scale_tensor = _get_jit_attribute(module, "scaler_scale")
    if mean_tensor is None or scale_tensor is None:
        raise RuntimeError("TorchScript module is missing scaler buffers.")

    feature_names = _get_jit_attribute(module, "feature_names") or []
    if isinstance(feature_names, torch.Tensor):
        feature_names = feature_names.tolist()

    return {
        "mean": _tensor_to_numpy(mean_tensor),
        "scale": _tensor_to_numpy(scale_tensor),
        "feature_names": list(feature_names)
        if hasattr(feature_names, "__iter__")
        else [],
    }


def _infer_model_dimensions_from_module(
    module: torch.jit.ScriptModule, scaler_params: dict[str, Any]
) -> dict[str, int]:
    """Use the scripted module to infer input, CV, and atom counts."""
    mean_data = scaler_params.get("mean")
    if mean_data is None:
        raise RuntimeError("Scaler parameters are required to infer model dimensions.")
    input_dim = int(np.asarray(mean_data).size)
    atom_count = _get_jit_attribute(module, "atom_count")
    atom_count = int(atom_count or input_dim)

    dummy_pos = torch.zeros(atom_count, 3, dtype=torch.float32)
    dummy_box = torch.eye(3, dtype=torch.float32)
    with torch.inference_mode():
        cvs = module.compute_cvs(dummy_pos, dummy_box)
    if cvs.ndim == 0:
        cv_dim = 1
    elif cvs.ndim == 1:
        cv_dim = int(cvs.shape[0])
    else:
        cv_dim = int(cvs.shape[-1])
    return {"input_dim": input_dim, "cv_dim": cv_dim, "atom_count": atom_count}


def _extract_feature_spec_hash_from_module(
    module: torch.jit.ScriptModule,
) -> str | None:
    hash_value = _get_jit_attribute(module, "feature_spec_sha256")
    if hash_value is None:
        return None
    if isinstance(hash_value, bytes):
        return hash_value.decode("utf-8", errors="ignore")
    return str(hash_value)


def _load_model_bundle(
    model_path: Path, spec_hash: str, feature_spec: Mapping[str, Any]
) -> tuple[Dict[str, Any], torch.jit.ScriptModule, int, int, Path]:
    if not model_path.exists():
        raise FileNotFoundError(f"CV model file not found: {model_path}")

    def _load_script_module(path: Path):
        info = load_cv_model_info(path.parent, path.stem)
        module = torch.jit.load(str(path), map_location="cpu")
        module.eval()
        return info, module

    torchscript_path = model_path
    bundle_info: Dict[str, Any]
    try:
        bundle_info, model = _load_script_module(torchscript_path)
    except RuntimeError as exc:
        if "constants.pkl" in str(exc):
            raise RuntimeError(
                "The provided CV model path appears to be a training checkpoint "
                "rather than an exported TorchScript bundle. Export the CV bias files "
                "with `python pmarlo_webapp/export_cv_bundle.py <model_base_path>` "
                "or pass the directory containing `deeptica_cv_model.pt`."
            ) from exc
        raise

    model_hash = bundle_info.get("feature_spec_sha256")
    if not model_hash:
        model_hash = _extract_feature_spec_hash_from_module(model)
        if model_hash is not None:
            bundle_info["feature_spec_sha256"] = model_hash
    if model_hash:
        if model_hash != spec_hash:
            raise RuntimeError(
                "Feature specification mismatch: "
                f"expected hash {spec_hash} from configuration but model provides {model_hash}."
            )
    else:
        logger.warning(
            "Exported CV model is missing feature_spec_sha256 metadata. "
            "Proceeding without strict feature-spec hash validation."
        )

    scaler_params = bundle_info.get("scaler_params", {})
    if not scaler_params:
        scaler_params = _extract_scaler_from_script_module(model)
        bundle_info["scaler_params"] = scaler_params

    mean = np.asarray(scaler_params.get("mean", []), dtype=np.float32)
    scale = np.asarray(scaler_params.get("scale", []), dtype=np.float32)
    ensure_scaler_finite(mean, scale)

    config_payload = dict(bundle_info.get("config", {}))
    expected_input_dim = int(config_payload.get("input_dim", 0))
    expected_cv_dim = int(config_payload.get("cv_dim", 0))
    expected_atom_count = int(config_payload.get("atom_count", 0))

    if expected_input_dim <= 0 or expected_cv_dim <= 0 or expected_atom_count <= 0:
        dimension_info = _infer_model_dimensions_from_module(model, scaler_params)
        expected_input_dim = expected_input_dim or dimension_info["input_dim"]
        expected_cv_dim = expected_cv_dim or dimension_info["cv_dim"]
        expected_atom_count = expected_atom_count or dimension_info["atom_count"]
        config_payload.setdefault("input_dim", expected_input_dim)
        config_payload.setdefault("cv_dim", expected_cv_dim)
        config_payload.setdefault("atom_count", expected_atom_count)

    if expected_input_dim <= 0 or expected_cv_dim <= 0 or expected_atom_count <= 0:
        raise RuntimeError("CV model configuration metadata is incomplete.")

    bundle_info["config"] = config_payload

    return bundle_info, model, expected_cv_dim, expected_atom_count, torchscript_path


def resolve_cv_model_torchscript_path(cv_model_path: str | Path) -> Path:
    """Resolve user-provided paths into a validated TorchScript CV bundle."""
    feature_spec, spec_hash = load_feature_spec()

    path = Path(cv_model_path).expanduser()
    try:
        path = path.resolve(strict=False)
    except Exception:
        path = path.absolute()

    candidate: Optional[Path] = None

    if path.is_dir():
        bundle = path / "deeptica_cv_model.pt"
        if bundle.exists():
            candidate = bundle
    elif path.suffix.lower() == ".pt" and path.name.endswith("_cv_model.pt"):
        candidate = path
    elif path.suffix.lower() == ".pt":
        bundle_dir = _find_existing_cv_bundle_dir(path)
        if bundle_dir is None:
            base = path.with_suffix("")
            raise FileNotFoundError(
                "No exported CV bundle could be located for checkpoint "
                f"{path}. Run `python pmarlo_webapp/export_cv_bundle.py {base}` "
                "or provide the directory that contains `deeptica_cv_model.pt`."
            )
        candidate = bundle_dir / "deeptica_cv_model.pt"

    if candidate is None or not candidate.exists():
        raise FileNotFoundError(
            f"Unable to locate a TorchScript CV model near {cv_model_path}. "
            "Provide a `deeptica_cv_model.pt` file or its containing directory."
        )

    _load_model_bundle(candidate, spec_hash, feature_spec)
    return candidate


def _validate_model_tensors(model: torch.jit.ScriptModule, spec_hash: str) -> None:
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            raise RuntimeError(
                f"Model parameter '{name}' must be float32 but is {param.dtype}."
            )
    for name, buffer in model.named_buffers():
        if buffer.dtype.is_floating_point and buffer.dtype != torch.float32:
            raise RuntimeError(
                f"Model buffer '{name}' must be float32 but is {buffer.dtype}."
            )

    # If the compiled TorchScript module exposes a feature_spec_sha256 attribute, ensure
    # it matches the configuration-derived spec_hash. If the attribute is missing (older
    # exports), do not fail—just log a warning and continue.
    attr_hash = getattr(model, "feature_spec_sha256", None)
    if attr_hash is None:
        logger.warning(
            "TorchScript module does not expose 'feature_spec_sha256' attribute. "
            "Skipping strict hash match; ensure your model bundle matches the active feature spec."
        )
    elif attr_hash != spec_hash:
        raise RuntimeError(
            "TorchScript module hash does not match configuration feature specification."
        )

    if not hasattr(model, "compute_cvs"):
        raise RuntimeError(
            "TorchScript CV model is missing the exported compute_cvs method required for monitoring."
        )


def _infer_cv_dimension(
    model: torch.jit.ScriptModule, expected_cv_dim: int, expected_atom_count: int
) -> tuple[int, bool]:
    dummy_pos = torch.zeros(expected_atom_count, 3, dtype=torch.float32)
    dummy_box = torch.eye(3, dtype=torch.float32)
    with torch.inference_mode():
        cvs = model.compute_cvs(dummy_pos, dummy_box)
    cv_dim = int(cvs.shape[-1]) if cvs.ndim > 1 else int(cvs.shape[0])
    if cv_dim != expected_cv_dim:
        raise RuntimeError(
            f"CV dimension mismatch: expected {expected_cv_dim}, observed {cv_dim}."
        )
    uses_pbc = bool(getattr(model, "uses_periodic_boundary_conditions", True))
    return cv_dim, uses_pbc


def _configure_torch_runtime(torch_threads: int) -> None:
    threads = max(1, int(torch_threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except AttributeError:
        pass


def _initialise_torch_force_legacy(
    model_path: Path, uses_pbc: bool, precision: str
):  # pragma: no cover - exercised indirectly in integration tests
    """Legacy approach: TorchForce with full model (slow on CPU).

    This is kept for backwards compatibility with old model exports.
    """
    from openmmtorch import TorchForce

    cv_force = TorchForce(str(model_path))
    cv_force.setForceGroup(1)
    cv_force.setUsesPeriodicBoundaryConditions(uses_pbc)
    try:
        cv_force.setProperty("precision", precision)
    except Exception as exc:  # pragma: no cover - exercised indirectly
        logger.warning(
            "TorchForce precision='%s' not accepted (OpenMM build may not expose the property); "
            "continuing with default precision. Error: %s",
            precision,
            exc,
        )
    return cv_force


def _initialise_optimized_cv_forces(
    model_path: Path,
    feature_spec: Dict[str, Any],
    system: openmm.System,
    uses_pbc: bool,
    precision: str,
) -> Tuple[List, List[int], Any]:
    """Optimized approach: OpenMM native features + NN-only TorchForce.

    Returns
    -------
    feature_forces : list
        OpenMM forces computing molecular features
    feature_indices : list of int
        Force indices in system
    torch_force : TorchForce or None
        NN-only bias force (None if NN model not available)
    """
    from pmarlo.features.deeptica.openmm_features import create_feature_forces

    # Create OpenMM native forces for feature computation (force group 2)
    feature_forces, feature_indices = create_feature_forces(
        feature_spec, system, force_group=2
    )

    logger.info(
        "Created %d OpenMM native feature forces (group 2) for fast computation",
        len(feature_forces),
    )

    # Check if NN-only model is available
    nn_model_path = Path(str(model_path).replace(".pt", "_nn.pt"))
    if not nn_model_path.exists():
        logger.warning(
            "NN-only model not found at %s. Falling back to legacy full model. "
            "Re-export your model with the updated export_cv_model() for optimal performance.",
            nn_model_path,
        )
        return feature_forces, feature_indices, None

    # NOTE: Current limitation - TorchForce can only take positions as input.
    # The NN-only model expects features, but we can't directly wire OpenMM forces to TorchForce.
    # For now, we compute features natively (which helps with monitoring) but still use
    # the full model in TorchForce for bias computation.
    # Future: Implement CustomCVForce wrapper or wait for openmmtorch to support feature input.

    logger.info(
        "NN-only model available at %s, but current openmmtorch API doesn't support "
        "feature-based input. Using native forces for monitoring only. "
        "Full model will still be used for bias computation.",
        nn_model_path,
    )

    return feature_forces, feature_indices, None


def _log_bias_configuration(
    *,
    bias_mode: str,
    torch_threads: int,
    precision: str,
    model_hash: str,
    spec_hash: str,
    force_group: int,
    uses_pbc: bool,
) -> None:
    logger.info("CV bias enabled (mode=%s)", bias_mode)
    logger.info("  Torch threads: %d", torch_threads)
    logger.info("  Torch precision: %s", precision)
    logger.info("  Model feature hash: %s", model_hash)
    logger.info("  Specification hash: %s", spec_hash)
    logger.info("  Force group: %d", force_group)
    logger.info("  Uses periodic boundary conditions: %s", uses_pbc)


def create_system(
    pdb: PDBFile,
    forcefield: ForceField,
    cv_model_path: str | None = None,
    cv_scaler_mean=None,
    cv_scaler_scale=None,
) -> openmm.System:
    """Create an OpenMM system with optional TorchForce-based CV biasing.

    See pmarlo_webapp/app/CV_INTEGRATION_GUIDE.md for usage guide.
    See pmarlo_webapp/app/CV_REQUIREMENTS.md for technical details.
    """

    config = _load_bias_config()
    system = _build_base_system(pdb, forcefield)

    if not config.enabled:
        _ensure_bias_disabled(cv_model_path)
        logger.info(
            "CV bias disabled via configuration; proceeding without TorchForce."
        )
        return system

    if cv_model_path is None:
        raise RuntimeError(
            "enable_cv_bias=true but no cv_model_path was provided. Supply a TorchScript bias model."
        )

    if config.mode != "harmonic":
        raise RuntimeError(
            f"Unsupported bias_mode '{config.mode}'. Only 'harmonic' biasing is currently available."
        )

    if not check_openmm_torch_available():
        raise RuntimeError(
            "openmm-torch is required to use CV biasing (install via `conda install -c conda-forge openmm-torch`)."
        )

    model_path = Path(cv_model_path).expanduser().resolve()
    feature_spec, spec_hash = load_feature_spec()
    (
        bundle_info,
        model,
        expected_cv_dim,
        expected_atom_count,
        torchscript_path,
    ) = _load_model_bundle(model_path, spec_hash, feature_spec)
    _validate_model_tensors(model, spec_hash)
    _, uses_pbc = _infer_cv_dimension(model, expected_cv_dim, expected_atom_count)

    _configure_torch_runtime(config.torch_threads)

    # Try to use optimized approach with native feature forces
    feature_spec = bundle_info.get("metadata", {}).get("feature_spec")
    feature_forces = []
    feature_indices = []

    if feature_spec is not None:
        try:
            feature_forces, feature_indices, nn_torch_force = (
                _initialise_optimized_cv_forces(
                    torchscript_path, feature_spec, system, uses_pbc, config.precision
                )
            )
            # Note: nn_torch_force is currently None due to API limitations
            # We still use legacy TorchForce for bias but have native forces for monitoring
        except Exception as exc:
            logger.warning(
                "Failed to initialize optimized CV forces: %s. Falling back to legacy approach.",
                exc,
            )
            feature_forces = []
            feature_indices = []

    # Always add the legacy TorchForce for bias computation (required until openmmtorch supports feature input)
    cv_force = _initialise_torch_force_legacy(
        torchscript_path, uses_pbc, config.precision
    )
    system.addForce(cv_force)

    # Store feature force information for monitoring (if available)
    if feature_forces:
        # Attach as metadata to the system for later retrieval by replica_exchange
        if not hasattr(system, "_pmarlo_feature_forces"):
            system._pmarlo_feature_forces = []  # type: ignore[attr-defined]
            system._pmarlo_feature_indices = []  # type: ignore[attr-defined]
        system._pmarlo_feature_forces = feature_forces  # type: ignore[attr-defined]
        system._pmarlo_feature_indices = feature_indices  # type: ignore[attr-defined]
        logger.info(
            "CV monitoring will use native OpenMM forces (2x speedup vs redundant PyTorch calls)"
        )

    model_hash = str(bundle_info.get("feature_spec_sha256"))
    _log_bias_configuration(
        bias_mode=config.mode,
        torch_threads=config.torch_threads,
        precision=config.precision,
        model_hash=model_hash,
        spec_hash=str(spec_hash),
        force_group=cv_force.getForceGroup(),
        uses_pbc=uses_pbc,
    )

    return system


def log_system_info(system: openmm.System, logger) -> None:
    logger.info("System created with %d particles", system.getNumParticles())
    logger.info("System has %d force terms", system.getNumForces())
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        logger.info("  Force %d: %s", idx, force.__class__.__name__)


def setup_metadynamics(
    system: openmm.System,
    bias_variables: Optional[List],
    reference_temperature_k: float,
    output_dir: Path,
):
    if not bias_variables:
        return None
    from openmm.app.metadynamics import Metadynamics

    bias_dir = output_dir / "bias"
    bias_dir.mkdir(exist_ok=True)
    meta = Metadynamics(
        system,
        bias_variables,
        temperature=reference_temperature_k * unit.kelvin,
        biasFactor=10.0,
        height=1.0 * unit.kilojoules_per_mole,
        frequency=500,
        biasDir=str(bias_dir),
        saveFrequency=1000,
    )
    return meta
