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

from pmarlo import api

# Compatibility shim for OpenMM XML deserialization API changes
if not hasattr(openmm.XmlSerializer, "load"):
    # Older OpenMM releases expose ``deserialize`` instead of ``load``.
    # Provide a small alias so downstream code can rely on ``load``
    # regardless of the installed OpenMM version.
    openmm.XmlSerializer.load = openmm.XmlSerializer.deserialize  # type: ignore[attr-defined]

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
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt

from pmarlo.replica_exchange.platform_selector import select_platform_and_properties
from pmarlo.replica_exchange.system_builder import create_system
from pmarlo.utils.integrator import create_langevin_integrator
from pmarlo.utils.progress import ProgressPrinter
from pmarlo.utils.seed import set_global_seed

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
        random_seed: Optional[int] = None,
        random_state: int | None = None,
    ):
        """
        Initialize the Simulation.

        Args:
            pdb_file: Path to the prepared PDB file
            temperature: Simulation temperature in Kelvin
            steps: Number of production steps
            output_dir: Directory for output files
            use_metadynamics: Whether to use metadynamics biasing
            random_seed: Seed for deterministic integrator behaviour. Deprecated,
                use ``random_state``.
            random_state: Seed for deterministic integrator behaviour.
        """
        self.pdb_file = pdb_file
        self.temperature = temperature
        self.steps = steps
        self.output_dir = Path(output_dir)
        self.use_metadynamics = use_metadynamics
        self.dcd_stride = dcd_stride
        self.random_seed = random_state if random_state is not None else random_seed
        if self.random_seed is not None:
            set_global_seed(int(self.random_seed))

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
            self.pdb_file,
            self.temperature,
            self.use_metadynamics,
            self.output_dir,
            self.random_seed,
        )
        return simulation, meta

    def run_production(self, openmm_simulation=None, meta=None) -> str:
        """Run production molecular dynamics simulation."""
        if openmm_simulation is None:
            openmm_simulation = self.openmm_simulation
        if meta is None:
            meta = self.meta

        # Make this Simulation instance discoverable by reporters/utilities
        try:
            setattr(openmm_simulation, "_owner", self)
        except Exception:
            pass

        trajectory_file = production_run(
            self.steps, openmm_simulation, meta, self.output_dir
        )
        return str(trajectory_file)

    def extract_features(
        self,
        trajectory_file: str,
        feature_specs: Sequence[str] | None = None,
        n_states: int = 40,
    ) -> np.ndarray:
        """Extract features from trajectory for MSM analysis."""
        states = feature_extraction(
            trajectory_file,
            self.pdb_file,
            random_state=self.random_seed,
            feature_specs=feature_specs,
            n_states=n_states,
        )
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
    random_seed: Optional[int] = None,
    random_state: int | None = None,
) -> Tuple["openmm.app.Simulation", Optional["Metadynamics"]]:
    """Prepare the molecular system with forcefield and optional metadynamics.

    Parameters
    ----------
    pdb_file_name:
        Path to the PDB structure.
    temperature:
        Simulation temperature in Kelvin.
    use_metadynamics:
        Whether to include a metadynamics bias.
    output_dir:
        Optional output directory for intermediate files.
    random_seed, random_state:
        Seeds for the OpenMM integrator. ``random_state`` takes precedence.
    """
    pdb = _load_pdb(pdb_file_name)
    forcefield = _create_forcefield()
    system = create_system(pdb, forcefield)
    meta = _maybe_create_metadynamics(
        system, pdb_file_name, temperature, use_metadynamics, output_dir
    )
    seed = random_state if random_state is not None else random_seed
    integrator = create_langevin_integrator(temperature, seed)
    platform, platform_properties = select_platform_and_properties(logger)
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


def _maybe_create_metadynamics(
    system: openmm.System,
    pdb_file_name: str,
    temperature: float,
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
        temperature=temperature_quantity(temperature),
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
    # Expose owner to allow reporters to access stride settings
    try:
        setattr(simulation, "_owner", None)
    except Exception:
        pass
    return simulation


def _minimize_and_equilibrate(simulation: app.Simulation) -> None:
    simulation.minimizeEnergy(maxIterations=100)
    simulation.step(1000)


def _print_production_stage_start() -> None:
    print("Stage 3/5  –  production run...")


def _resolve_output_dir(output_dir: Optional[Path]) -> Path:
    return Path("output") / "simulation" if output_dir is None else Path(output_dir)


def _compose_dcd_filename(output_dir: Path) -> str:
    return str(output_dir / "traj.dcd")


def _determine_dcd_stride(simulation: app.Simulation) -> int:
    try:
        stride = getattr(getattr(simulation, "_owner", None), "dcd_stride", 1000)
        if not isinstance(stride, int) or stride <= 0:
            return 1000
        return int(stride)
    except Exception:
        return 1000


def _attach_dcd_reporter(
    simulation: app.Simulation, dcd_filename: str, stride: int
) -> app.DCDReporter:
    reporter = app.DCDReporter(dcd_filename, int(max(1, stride)))
    simulation.reporters.append(reporter)
    return reporter


def _run_metadynamics(
    meta: Metadynamics, simulation: app.Simulation, total_steps: int, step_size: int
) -> list[float]:
    bias_list: list[float] = []
    full_chunks = total_steps // step_size
    remainder = total_steps % step_size

    for _ in range(full_chunks):
        meta.step(simulation, step_size)
        simulation.step(0)
        try:
            bias_val = meta._currentBias  # type: ignore[attr-defined]
        except AttributeError:
            bias_val = 0.0
        for _ in range(step_size):
            bias_list.append(float(bias_val))

    if remainder:
        meta.step(simulation, remainder)
        simulation.step(0)
        try:
            bias_val = meta._currentBias  # type: ignore[attr-defined]
        except AttributeError:
            bias_val = 0.0
        for _ in range(remainder):
            bias_list.append(float(bias_val))
    return bias_list


def _run_plain_md(simulation: app.Simulation, total_steps: int) -> None:
    simulation.step(total_steps)


def _save_final_state(simulation: app.Simulation, output_dir: Path) -> None:
    simulation.saveState(str(output_dir / "final.xml"))


def _cleanup_dcd(simulation: app.Simulation, reporter: app.DCDReporter) -> None:
    simulation.reporters.remove(reporter)
    import gc

    del reporter
    gc.collect()


def _save_bias_if_present(
    meta: Optional[Metadynamics], bias_list: list[float], output_dir: Path
) -> None:
    if meta is None:
        return
    bias_array = np.array(bias_list)
    bias_file = output_dir / "bias_for_run.npy"
    np.save(str(bias_file), bias_array)
    print(
        f"[INFO] Saved bias array for this run to {bias_file} (length: {len(bias_array)})"
    )


def production_run(steps, simulation, meta, output_dir=None):
    """Run production molecular dynamics simulation."""
    _print_production_stage_start()

    out_dir = _resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dcd_filename = _compose_dcd_filename(out_dir)
    stride = _determine_dcd_stride(simulation)
    dcd_reporter = _attach_dcd_reporter(simulation, dcd_filename, stride)

    total_steps = int(steps)
    step_size = 10

    progress = ProgressPrinter(total_steps)

    if meta is not None:
        # Metadynamics runs in chunks; update progress after every chunk
        full_chunks = total_steps // step_size
        remainder = total_steps % step_size
        bias_list: list[float] = []

        done = 0
        for _ in range(full_chunks):
            part = _run_metadynamics(meta, simulation, step_size, step_size)
            bias_list.extend(part)
            done += step_size
            progress.draw(done)
            progress.newline_if_active()
        if remainder:
            part = _run_metadynamics(meta, simulation, remainder, remainder)
            bias_list.extend(part)
            done += remainder
            progress.draw(done)
            progress.newline_if_active()
    else:
        # Plain MD – step in chunks to surface progress
        full_chunks = total_steps // step_size
        remainder = total_steps % step_size
        done = 0
        for _ in range(full_chunks):
            simulation.step(step_size)
            done += step_size
            progress.draw(done)
            progress.newline_if_active()
        if remainder:
            simulation.step(remainder)
            done += remainder
            progress.draw(done)
            progress.newline_if_active()
        bias_list = []

    progress.close()

    _save_final_state(simulation, out_dir)
    print("✔ MD + biasing finished\n")

    _cleanup_dcd(simulation, dcd_reporter)
    _save_bias_if_present(meta, bias_list, out_dir)

    return dcd_filename


def feature_extraction(
    dcd_path,
    pdb_path,
    random_state: int | None = 0,
    feature_specs: Sequence[str] | None = None,
    n_states: int = 40,
):
    """Extract features from trajectory for MSM analysis.

    Parameters
    ----------
    dcd_path:
        Path to the trajectory file in DCD format.
    pdb_path:
        Path to the corresponding PDB topology file.
    random_state:
        Seed for deterministic clustering.  When ``None`` a random seed is
        used, otherwise the provided seed ensures reproducible clustering.
        Defaults to ``0`` for backward compatibility with earlier releases.
    feature_specs:
        Optional sequence of feature specifications passed to
        :func:`pmarlo.api.compute_features`.  Defaults to ``["phi_psi"]``.
    n_states:
        Number of microstates to identify during clustering.
    """
    print("Stage 4/5  –  featurisation + clustering ...")

    traj = md.load(dcd_path, top=pdb_path)
    print("Number of frames loaded:", traj.n_frames)

    specs = feature_specs if feature_specs is not None else ["phi_psi"]
    X, _cols, _periodic = api.compute_features(traj, feature_specs=specs)

    states = api.cluster_microstates(
        X,
        method="minibatchkmeans",
        n_states=n_states,
        random_state=random_state,
    )
    print("✔ Featurisation + clustering done\n")
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
