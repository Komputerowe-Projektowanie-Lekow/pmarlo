from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _load_model_bundle(
    model_path: Path, spec_hash: str
) -> tuple[Dict[str, Any], torch.jit.ScriptModule, int, int]:
    if not model_path.exists():
        raise FileNotFoundError(f"CV model file not found: {model_path}")

    bundle_info = load_cv_model_info(model_path.parent, model_path.stem)

    model_hash = bundle_info.get("feature_spec_sha256")
    if not model_hash:
        raise RuntimeError("Exported CV model is missing feature_spec_sha256 metadata.")
    if model_hash != spec_hash:
        raise RuntimeError(
            "Feature specification mismatch: "
            f"expected hash {spec_hash} from configuration but model provides {model_hash}."
        )

    scaler_params = bundle_info.get("scaler_params", {})
    mean = np.asarray(scaler_params.get("mean", []), dtype=np.float32)
    scale = np.asarray(scaler_params.get("scale", []), dtype=np.float32)
    ensure_scaler_finite(mean, scale)

    config_payload = bundle_info.get("config", {})
    expected_input_dim = int(config_payload.get("input_dim", 0))
    expected_cv_dim = int(config_payload.get("cv_dim", 0))
    expected_atom_count = int(config_payload.get("atom_count", 0))
    if expected_input_dim <= 0 or expected_cv_dim <= 0 or expected_atom_count <= 0:
        raise RuntimeError("CV model configuration metadata is incomplete.")

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()

    return bundle_info, model, expected_cv_dim, expected_atom_count


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

    attr_hash = getattr(model, "feature_spec_sha256", None)
    if attr_hash != spec_hash:
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


def _initialise_torch_force(
    model_path: Path, uses_pbc: bool, precision: str
):  # pragma: no cover - exercised indirectly in integration tests
    from openmmtorch import TorchForce

    cv_force = TorchForce(str(model_path))
    cv_force.setForceGroup(1)
    cv_force.setUsesPeriodicBoundaryConditions(uses_pbc)
    try:
        cv_force.setProperty("precision", precision)
    except AttributeError as exc:
        raise RuntimeError(
            f"TorchForce does not support precision='{precision}' on this platform."
        ) from exc
    return cv_force


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

    See example_programs/app_usecase/app/CV_INTEGRATION_GUIDE.md for usage guide.
    See example_programs/app_usecase/app/CV_REQUIREMENTS.md for technical details.
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
    _, spec_hash = load_feature_spec()
    bundle_info, model, expected_cv_dim, expected_atom_count = _load_model_bundle(
        model_path, spec_hash
    )
    _validate_model_tensors(model, spec_hash)
    _, uses_pbc = _infer_cv_dimension(model, expected_cv_dim, expected_atom_count)

    _configure_torch_runtime(config.torch_threads)
    cv_force = _initialise_torch_force(model_path, uses_pbc, config.precision)
    system.addForce(cv_force)

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
