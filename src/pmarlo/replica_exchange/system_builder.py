from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import openmm
from openmm import unit
from openmm.app import PME, ForceField, HBonds, PDBFile

from pmarlo import constants as const


def load_pdb_and_forcefield(
    pdb_file: str, forcefield_files: List[str]
) -> Tuple[PDBFile, ForceField]:
    pdb = PDBFile(pdb_file)
    forcefield = ForceField(*forcefield_files)
    return pdb, forcefield


def create_system(
    pdb: PDBFile,
    forcefield: ForceField,
    cv_model_path: str | None = None,
    cv_scaler_mean=None,
    cv_scaler_scale=None,
) -> openmm.System:
    """Create an OpenMM system with optional TorchForce-based CV biasing."""

    import logging

    logger = logging.getLogger(__name__)

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

    if cv_model_path is not None:
        try:
            from pmarlo.features.deeptica import check_openmm_torch_available

            if not check_openmm_torch_available():
                raise RuntimeError(
                    "openmm-torch is required to use CV biasing "
                    "(install via `conda install -c conda-forge openmm-torch`)."
                )

            import torch
            from openmmtorch import TorchForce

            torch_threads = int(os.environ.get("PMARLO_TORCH_THREADS", "4"))
            torch.set_num_threads(max(1, torch_threads))
            try:
                torch.set_num_interop_threads(max(1, torch_threads))
            except AttributeError:
                pass

            logger.info("Loading TorchScript CV bias module from %s", cv_model_path)
            model = torch.jit.load(str(cv_model_path), map_location="cpu")
            model.eval()

            feature_hash = getattr(model, "feature_spec_sha256", None)
            if feature_hash is None:
                raise RuntimeError(
                    "TorchScript CV module is missing feature_spec hash metadata"
                )

            uses_pbc = bool(getattr(model, "uses_periodic_boundary_conditions", True))

            cv_force = TorchForce(str(cv_model_path))
            cv_force.setForceGroup(1)
            cv_force.setUsesPeriodicBoundaryConditions(uses_pbc)
            try:
                cv_force.setProperty("precision", "single")
            except AttributeError:
                pass

            system.addForce(cv_force)

            logger.info(
                "Added CV bias potential (feature hash=%s, torch threads=%d, uses_pbc=%s)",
                feature_hash,
                max(1, torch_threads),
                uses_pbc,
            )
        except Exception as exc:
            logger.error("Failed to attach CV bias potential: %s", exc, exc_info=True)
            raise

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
