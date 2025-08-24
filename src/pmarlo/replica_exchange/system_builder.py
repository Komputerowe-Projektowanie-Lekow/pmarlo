from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import openmm
from openmm import unit
from openmm.app import PME, ForceField, HBonds, PDBFile


def load_pdb_and_forcefield(
    pdb_file: str, forcefield_files: List[str]
) -> Tuple[PDBFile, ForceField]:
    pdb = PDBFile(pdb_file)
    forcefield = ForceField(*forcefield_files)
    return pdb, forcefield


def create_system(pdb: PDBFile, forcefield: ForceField) -> openmm.System:
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        constraints=HBonds,
        rigidWater=True,
        nonbondedCutoff=0.9 * unit.nanometer,
        ewaldErrorTolerance=1e-4,
        hydrogenMass=3.0 * unit.amu,
    )
    # Avoid duplicate CMMotionRemover if ForceField already inserted one
    try:
        has_cmm = any(
            isinstance(system.getForce(i), openmm.CMMotionRemover)
            for i in range(system.getNumForces())
        )
        if not has_cmm:
            system.addForce(openmm.CMMotionRemover())
    except Exception:
        pass
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
