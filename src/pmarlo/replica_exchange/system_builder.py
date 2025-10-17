from __future__ import annotations

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
    """
    Create OpenMM system with optional CV-based biasing.

    Parameters
    ----------
    pdb : PDBFile
        Input PDB structure
    forcefield : ForceField
        OpenMM force field
    cv_model_path : str, optional
        Path to TorchScript CV model for biased sampling
    cv_scaler_mean : np.ndarray, optional
        Scaler mean for CV model
    cv_scaler_scale : np.ndarray, optional
        Scaler scale for CV model

    Returns
    -------
    openmm.System
        Configured OpenMM system
    """
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
    # Avoid duplicate CMMotionRemover if ForceField already inserted one
    has_cmm = any(
        isinstance(system.getForce(i), openmm.CMMotionRemover)
        for i in range(system.getNumForces())
    )
    if not has_cmm:
        system.addForce(openmm.CMMotionRemover())

    # Add CV-based biasing if model is provided
    if cv_model_path is not None:
        try:
            from pmarlo.features.deeptica import (
                add_cv_bias_to_system,
                check_openmm_torch_available,
            )

            if not check_openmm_torch_available():
                logger.warning(
                    "CV model specified but openmm-torch not available. "
                    "Install with: conda install -c conda-forge openmm-torch. "
                    "Continuing without CV biasing."
                )
            elif cv_scaler_mean is not None and cv_scaler_scale is not None:
                logger.info("Adding CV-based biasing force from model: %s", cv_model_path)
                add_cv_bias_to_system(
                    system,
                    model_path=cv_model_path,
                    scaler_mean=cv_scaler_mean,
                    scaler_scale=cv_scaler_scale,
                    bias_strength=1.0,  # Can be parameterized if needed
                    force_group=1,  # Separate group for CV force
                )
                logger.info("CV biasing force successfully added to system")
            else:
                logger.warning("CV model path provided but scaler parameters missing. Skipping CV biasing.")

        except ImportError as exc:
            logger.warning(
                "Could not import CV integration modules: %s. "
                "Continuing without CV biasing.",
                exc
            )
        except Exception as exc:
            logger.error(
                "Failed to add CV biasing force: %s. "
                "Continuing without CV biasing.",
                exc
            )

    return system


def log_system_info(system: openmm.System, logger) -> None:
    logger.info(f"System created with {system.getNumParticles()} particles")
    logger.info(f"System has {system.getNumForces()} force terms")
    for force_idx in range(system.getNumForces()):
        force = system.getForce(force_idx)
        logger.info(f"  Force {force_idx}: {force.__class__.__name__}")


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
