# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Simulation module for PMARLO.

Provides molecular dynamics simulation capabilities with metadynamics and
system preparation.
"""

from collections import defaultdict

import mdtraj as md
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openmm.app.metadynamics import BiasVariable, Metadynamics
from sklearn.cluster import MiniBatchKMeans

# PDBFixer is optional - users can install with: pip install "pmarlo[fixer]"
try:
    from pdbfixer import PDBFixer

    HAS_PDBFIXER = True
except ImportError:
    PDBFixer = None
    HAS_PDBFIXER = False
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BASE_DIR / "tests" / "data"


class Simulation:
    """
    Molecular Dynamics Simulation class for PMARLO.

    Handles system preparation, equilibration, and production runs with metadynamics.
    """

    def __init__(
        self,
        pdb_file: str,
        temperature: float = 300.0,
        steps: int = 1000,
        output_dir: str = "output/simulation",
        use_metadynamics: bool = True,
        dcd_stride: int = 1000,
    ):
        """
        Initialize the Simulation.

        Args:
            pdb_file: Path to the prepared PDB file
            temperature: Simulation temperature in Kelvin
            steps: Number of production steps
            output_dir: Directory for output files
            use_metadynamics: Whether to use metadynamics biasing
        """
        self.pdb_file = pdb_file
        self.temperature = temperature
        self.steps = steps
        self.output_dir = Path(output_dir)
        self.use_metadynamics = use_metadynamics
        self.dcd_stride = dcd_stride

        # OpenMM objects
        self.openmm_simulation = None
        self.meta = None
        self.system = None
        self.integrator = None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_system(
        self,
    ) -> Tuple["openmm.app.Simulation", Optional["Metadynamics"]]:
        """Prepare the molecular system with forcefield and optional metadynamics."""
        simulation, meta = prepare_system(
            self.pdb_file, self.temperature, self.use_metadynamics, self.output_dir
        )
        return simulation, meta

    def run_production(self, openmm_simulation=None, meta=None) -> str:
        """Run production molecular dynamics simulation."""
        if openmm_simulation is None:
            openmm_simulation = self.openmm_simulation
        if meta is None:
            meta = self.meta

        trajectory_file = production_run(
            self.steps, openmm_simulation, meta, self.output_dir
        )
        return str(trajectory_file)

    def extract_features(self, trajectory_file: str) -> np.ndarray:
        """Extract features from trajectory for MSM analysis."""
        states = feature_extraction(trajectory_file, self.pdb_file)
        return np.array(states)

    def run_complete_simulation(self) -> Tuple[str, np.ndarray]:
        """Run complete simulation pipeline and return trajectory file and states."""
        logger.info(f"Starting simulation for {self.pdb_file}")

        # Prepare system
        self.openmm_simulation, self.meta = self.prepare_system()

        # Run production
        trajectory_file = self.run_production()

        # Extract features
        states = self.extract_features(trajectory_file)

        logger.info(f"Simulation complete. Trajectory: {trajectory_file}")
        return trajectory_file, states


def prepare_system(
    pdb_file_name: str,
    temperature: float = 300.0,
    use_metadynamics: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple["openmm.app.Simulation", Optional["Metadynamics"]]:
    """Prepare the molecular system with forcefield and optional metadynamics."""
    pdb = _load_pdb(pdb_file_name)
    forcefield = _create_forcefield()
    system = _create_system(pdb, forcefield)
    meta = _maybe_create_metadynamics(
        system, pdb_file_name, use_metadynamics, output_dir
    )
    integrator = _create_integrator(temperature)
    platform, platform_properties = _select_platform()
    simulation = _create_openmm_simulation(
        pdb, system, integrator, platform, platform_properties
    )
    _minimize_and_equilibrate(simulation)
    print("✔ Build & equilibration complete\n")
    return simulation, meta


# -------------------------- Helper functions --------------------------


def _load_pdb(pdb_file_name: str) -> app.PDBFile:
    return app.PDBFile(pdb_file_name)


def _create_forcefield() -> app.ForceField:
    return app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")


def _create_system(pdb: app.PDBFile, forcefield: app.ForceField) -> openmm.System:
    return forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        constraints=app.HBonds,
        rigidWater=True,
        nonbondedCutoff=unit.Quantity(0.9, unit.nanometer),
        ewaldErrorTolerance=1e-4,
        hydrogenMass=unit.Quantity(3.0, unit.amu),  # HMR
        removeCMMotion=True,
    )


def _maybe_create_metadynamics(
    system: openmm.System,
    pdb_file_name: str,
    use_metadynamics: bool,
    output_dir: Optional[Path],
) -> Optional[Metadynamics]:
    if not use_metadynamics:
        return None
    traj0 = md.load_pdb(pdb_file_name)
    phi_indices, _ = md.compute_phi(traj0)
    if len(phi_indices) == 0:
        raise RuntimeError(
            "No φ dihedral found in the PDB structure – cannot set up CV."
        )
    phi_atoms = [int(i) for i in phi_indices[0]]
    phi_force = openmm.CustomTorsionForce("theta")
    phi_force.addTorsion(*phi_atoms, [])
    phi_cv = BiasVariable(
        phi_force,
        minValue=-np.pi,
        maxValue=np.pi,
        biasWidth=0.35,  # ~20°
        periodic=True,
    )
    bias_dir = _ensure_bias_dir(output_dir)
    _clear_existing_bias_files(bias_dir)
    return Metadynamics(
        system,
        [phi_cv],
        temperature=temperature_quantity(300.0),
        biasFactor=10.0,
        height=1.0 * unit.kilojoules_per_mole,
        frequency=500,  # hill every 1 ps (500 × 2 fs)
        biasDir=str(bias_dir),
        saveFrequency=1000,
    )


def temperature_quantity(value_kelvin: float) -> unit.Quantity:
    return value_kelvin * unit.kelvin


def _ensure_bias_dir(output_dir: Optional[Path]) -> Path:
    if output_dir is None:
        base = Path("output") / "simulation"
    else:
        base = Path(output_dir)
    bias_dir = base / "bias"
    os.makedirs(str(bias_dir), exist_ok=True)
    return bias_dir


def _clear_existing_bias_files(bias_dir: Path) -> None:
    if bias_dir.exists():
        for file in bias_dir.glob("bias_*.npy"):
            try:
                file.unlink()
            except Exception:
                pass


def _create_integrator(temperature: float) -> openmm.Integrator:
    return openmm.LangevinIntegrator(
        temperature * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds
    )


def _select_platform() -> Tuple[openmm.Platform, dict]:
    platform_properties: dict = {}
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform_properties = {
            "Precision": "mixed",
            "UseFastMath": "true",
            "DeterministicForces": "false",
        }
        logger.info("Using CUDA (mixed precision, fast math)")
    except Exception:
        try:
            try:
                platform = openmm.Platform.getPlatformByName("HIP")
                logger.info("Using HIP (AMD GPU)")
            except Exception:
                platform = openmm.Platform.getPlatformByName("OpenCL")
                logger.info("Using OpenCL")
        except Exception:
            platform = openmm.Platform.getPlatformByName("CPU")
            try:
                openmm.Platform.setPropertyDefaultValue(
                    "CpuThreads", str(os.cpu_count() or 1)
                )
            except Exception:
                pass
            logger.info("Using CPU with all cores")
    return platform, platform_properties


def _create_openmm_simulation(
    pdb: app.PDBFile,
    system: openmm.System,
    integrator: openmm.Integrator,
    platform: openmm.Platform,
    platform_properties: Optional[dict],
) -> app.Simulation:
    simulation = app.Simulation(
        pdb.topology, system, integrator, platform, platform_properties or None
    )
    simulation.context.setPositions(pdb.positions)
    return simulation


def _minimize_and_equilibrate(simulation: app.Simulation) -> None:
    simulation.minimizeEnergy(maxIterations=100)
    simulation.step(1000)


def production_run(steps, simulation, meta, output_dir=None):
    """Run production molecular dynamics simulation."""
    print("Stage 3/5  –  production run...")

    if output_dir is None:
        output_dir = Path("output") / "simulation"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dcd_filename = str(output_dir / "traj.dcd")
    # Respect Simulation.dcd_stride if available via bound simulation.owner;
    # default 1000
    try:
        stride = getattr(getattr(simulation, "_owner", None), "dcd_stride", 1000)
    except Exception:
        stride = 1000
    dcd = app.DCDReporter(dcd_filename, int(max(1, stride)))
    simulation.reporters.append(dcd)

    total_steps = steps
    step_size = 10
    bias_list = []

    if meta is not None:
        for i in range(total_steps // step_size):
            meta.step(simulation, step_size)
            simulation.step(0)  # triggers reporters
            try:
                bias_val = meta._currentBias
            except AttributeError:
                bias_val = 0.0
            for _ in range(step_size):
                bias_list.append(bias_val)
    else:
        # Run without metadynamics
        simulation.step(total_steps)

    simulation.saveState(str(output_dir / "final.xml"))
    print("✔ MD + biasing finished\n")

    # Remove DCDReporter and force garbage collection to finalize file
    simulation.reporters.remove(dcd)
    import gc

    del dcd
    gc.collect()

    if meta is not None:
        # Save the bias array for this run
        bias_array = np.array(bias_list)
        bias_file = output_dir / "bias_for_run.npy"
        np.save(str(bias_file), bias_array)
        print(
            f"[INFO] Saved bias array for this run to {bias_file} "
            f"(length: {len(bias_array)})"
        )

    return dcd_filename


def feature_extraction(dcd_path, pdb_path):
    """Extract features from trajectory for MSM analysis."""
    print("Stage 4/5  –  featurisation + clustering ...")

    # Load the trajectory and compute φ dihedral angles
    t = md.load(dcd_path, top=pdb_path)
    print("Number of frames loaded:", t.n_frames)
    phi_vals, _ = md.compute_phi(t)
    phi_vals = phi_vals.squeeze()
    X = np.cos(phi_vals)
    X = X.reshape(-1, 1)

    kmeans = MiniBatchKMeans(n_clusters=40, random_state=0).fit(X)
    states = kmeans.labels_
    print("✔ Clustering done\n")
    return states


def build_transition_model(states, bias=None):
    """Build transition model from clustered states."""
    print("Stage 5/5  –  Markov model ...")

    tau = 20  # frames → 40 ps
    C = defaultdict(float)
    kT = 0.593  # kcal/mol at 300K
    F_est = 0.0  # For now, can be improved later
    n_transitions = len(states) - tau
    if bias is not None and len(bias) != len(states):
        raise ValueError(
            f"Bias array length ({len(bias)}) does not match number of states "
            f"({len(states)})"
        )
    for i in range(n_transitions):
        if bias is not None:
            w_t = np.exp((bias[i] - F_est) / kT)
        else:
            w_t = 1.0
        C[(states[i], states[i + tau])] += w_t

    # Dense count matrix → row-normalised transition matrix
    n = np.max(states) + 1
    Cmat = np.zeros((n, n))
    for (i, j), w in C.items():
        Cmat[i, j] = w

    T = (Cmat.T / Cmat.sum(axis=1)).T  # row-stochastic

    # Stationary distribution (left eigenvector of T)
    evals, evecs = np.linalg.eig(T.T)
    pi = np.real_if_close(evecs[:, np.argmax(evals)].flatten())
    pi /= pi.sum()
    DG = -kT * np.log(pi)  # 0.593 kcal/mol ≈ kT at 300 K

    print("✔ Finished – free energies (kcal/mol) written to DG array")
    return DG


def relative_energies(DG):
    """Calculate relative energies."""
    return DG - np.min(DG)


def plot_DG(DG):
    """Plot free energy profile."""
    plt.figure()
    plt.bar(np.arange(len(DG)), DG, color="blue")
    plt.xlabel("State Index")
    plt.ylabel("Free Energy (kcal/mol)")
    plt.title("Free Energy Profile")
    plt.tight_layout()
    plt.show()
