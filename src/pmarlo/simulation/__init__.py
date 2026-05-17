"""OpenMM-based MD segment runner for adaptive sampling pipelines."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

logger = logging.getLogger(__name__)

__all__ = ["run_segment"]

if TYPE_CHECKING:
    import mdtraj


def run_segment(
    pdb_path: Union[str, Path],
    n_steps: int = 10_000,
    temperature_K: float = 300.0,
    timestep_fs: float = 2.0,
    output_path: Optional[Union[str, Path]] = None,
    force_field: str = "amber14-all.xml",
    implicit_solvent: str = "implicit/gbn2.xml",
    report_interval: int = 100,
    platform_name: str = "CUDA",
) -> "mdtraj.Trajectory":
    """Run a short OpenMM MD segment and return the trajectory.

    Parameters
    ----------
    pdb_path:
        Input PDB file path.
    n_steps:
        Number of integration steps.
    temperature_K:
        Simulation temperature in Kelvin.
    timestep_fs:
        Integration timestep in femtoseconds.
    output_path:
        Where to write the DCD file. If None a temporary file is used.
    force_field:
        AMBER force-field XML to use.
    implicit_solvent:
        Implicit-solvent XML to use (GBN2 by default, no water box required).
    report_interval:
        Write one frame every this many steps.
    platform_name:
        OpenMM platform ("CUDA", "OpenCL", "CPU", "Reference").
        Falls back to CPU if the requested platform is unavailable.

    Returns
    -------
    mdtraj.Trajectory
        The simulated trajectory loaded in memory.
    """
    import mdtraj
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    pdb_path = Path(pdb_path)
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(force_field, implicit_solvent)
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        hydrogenMass=1.5 * unit.amu,
    )

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )

    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        properties: dict = (
            {"Precision": "mixed"} if platform_name in ("CUDA", "OpenCL") else {}
        )
    except Exception:
        logger.warning("Platform %s unavailable, falling back to CPU.", platform_name)
        platform = openmm.Platform.getPlatformByName("CPU")
        properties = {}

    simulation = app.Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=100)
    simulation.context.setVelocitiesToTemperature(temperature_K * unit.kelvin)

    if output_path is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="pmarlo_seg_"))
        dcd_path = tmp_dir / "traj.dcd"
    else:
        dcd_path = Path(output_path)
        dcd_path.parent.mkdir(parents=True, exist_ok=True)

    simulation.reporters.append(app.DCDReporter(str(dcd_path), report_interval))
    logger.info(
        "Running %d steps at %.1f K on %s (dt=%.1f fs)",
        n_steps,
        temperature_K,
        platform_name,
        timestep_fs,
    )
    simulation.step(n_steps)
    logger.info("Segment done → %s", dcd_path)

    return mdtraj.load(str(dcd_path), top=str(pdb_path))
