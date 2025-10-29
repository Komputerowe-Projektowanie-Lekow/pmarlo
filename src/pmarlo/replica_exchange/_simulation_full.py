# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 PMARLO Development Team

"""
Simulation module for PMARLO.

Provides molecular dynamics simulation capabilities with metadynamics and
system preparation.
"""

import logging
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import mdtraj as md
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openmm.app.metadynamics import BiasVariable, Metadynamics

if not hasattr(openmm.XmlSerializer, "load"):  # pragma: no cover - compatibility
    openmm.XmlSerializer.load = staticmethod(openmm.XmlSerializer.deserialize)

from pmarlo import api

from .bias_hook import BiasHook

# PDBFixer is optional - users can install with: pip install "pmarlo[fixer]"
try:
    import pdbfixer
except ImportError:
    pdbfixer = None


class Simulation:
    """
    Molecular dynamics simulation manager for PMARLO.

    Provides high-level interface for system preparation, simulation execution,
    and analysis. Supports both standard MD and enhanced sampling methods like
    metadynamics.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file for the system
    output_dir : str, optional
        Directory for output files (default: "output")
    temperature : float, optional
        Simulation temperature in Kelvin (default: 300.0)
    pressure : float, optional
        Simulation pressure in bar (default: 1.0)
    platform : str, optional
        OpenMM platform to use ("CUDA" for GPU or "CPU" for CPU). The value must be
        provided explicitly—no automatic fallback or heuristic selection is
        performed.
    steps : int, optional
        Default number of production steps when :meth:`run_production` is called
        without an explicit value.  Defaults to 100000 for backwards compatibility.
    use_metadynamics : bool, optional
        Whether metadynamics biases should be configured during preparation
        (default: True).
    random_seed : int, optional
        Seed forwarded to OpenMM components for deterministic trajectories.
    """

    def __init__(
        self,
        pdb_file: str,
        output_dir: str = "output",
        temperature: float = 300.0,
        pressure: float = 1.0,
        platform: str = "CUDA",
        *,
        steps: int | None = None,
        use_metadynamics: bool = True,
        random_seed: int | None = None,
    ):
        self.pdb_file = str(pdb_file)
        self.output_dir = Path(output_dir or "output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = float(temperature)
        self.pressure = float(pressure)
        self.platform_name = platform
        self.steps = int(steps) if steps is not None else 100000
        self.use_metadynamics = bool(use_metadynamics)
        self.random_seed = random_seed

        # Initialize OpenMM objects
        self.pdb = None
        self.forcefield = None
        self.system = None
        self.simulation = None
        self.openmm_simulation = None
        self.platform = None

        # Trajectory storage
        self.trajectory_data: list[Any] = []
        self.energies: dict[str, list[Any]] = defaultdict(list)

        # Metadynamics setup
        self.metadynamics = None
        self.bias_variables: list[Any] = []
        self.bias_hook: Optional[BiasHook] = None
        self.meta = None

    @property
    def temperature_quantity(self) -> unit.Quantity:
        """OpenMM-compatible temperature quantity."""

        return self.temperature * unit.kelvin

    @property
    def pressure_quantity(self) -> unit.Quantity:
        """OpenMM-compatible pressure quantity."""

        return self.pressure * unit.bar

    def prepare_system(self, forcefield_files=None, water_model="tip3p"):
        """
        Prepare the molecular system for simulation.

        Parameters
        ----------
        forcefield_files : list, optional
            Force field XML files to use
        water_model : str, optional
            Water model to use (default: "tip3p")
        Returns
        -------
        tuple[app.Simulation, Metadynamics | None]
            Prepared OpenMM simulation and optional metadynamics driver.
        """
        if forcefield_files is None:
            forcefield_files = ["amber14-all.xml", f"{water_model}.xml"]

        # Load PDB file
        self.pdb = app.PDBFile(self.pdb_file)

        # Optional: Fix common PDB issues
        if pdbfixer is not None:
            self._fix_pdb_issues()

        # Load force field
        self.forcefield = app.ForceField(*forcefield_files)

        # Create system
        self.system = self.forcefield.createSystem(
            self.pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=app.HBonds,
        )

        # Add barostat for NPT
        barostat = openmm.MonteCarloBarostat(
            self.pressure_quantity, self.temperature_quantity
        )
        if self.random_seed is not None:
            barostat.setRandomNumberSeed(int(self.random_seed))
        self.system.addForce(barostat)

        # Set up platform
        self._setup_platform()

        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature_quantity, 1 / unit.picosecond, 0.002 * unit.picoseconds
        )
        if self.random_seed is not None:
            integrator.setRandomNumberSeed(int(self.random_seed))

        self.simulation = app.Simulation(
            self.pdb.topology, self.system, integrator, self.platform
        )
        self.simulation.context.setPositions(self.pdb.positions)
        if self.random_seed is not None:
            self.simulation.context.setVelocitiesToTemperature(
                self.temperature_quantity, self.random_seed
            )

        self.openmm_simulation = self.simulation

        if self.use_metadynamics and self.bias_variables:
            bias_dir = self.output_dir / "bias"
            bias_dir.mkdir(parents=True, exist_ok=True)
            self.metadynamics = Metadynamics(
                self.system,
                self.bias_variables,
                self.temperature_quantity,
                biasFactor=10,
                height=1.0 * unit.kilojoules_per_mole,
                frequency=500,
                biasDir=str(bias_dir),
            )
        else:
            self.metadynamics = None

        self.meta = self.metadynamics
        return self.openmm_simulation, self.meta

    def _fix_pdb_issues(self):
        """Fix common PDB issues using PDBFixer."""
        if pdbfixer is None:
            return

        fixer = pdbfixer.PDBFixer(pdb=self.pdb)

        # Find and add missing residues
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        # Add missing hydrogens
        fixer.addMissingHydrogens(7.0)

        # Update PDB object
        self.pdb = fixer

    def _setup_platform(self):
        """Set up the OpenMM platform using the explicitly requested backend."""
        supported_platforms = {"CPU", "CUDA"}
        if self.platform_name not in supported_platforms:
            raise ValueError(
                "Unsupported OpenMM platform '%s'. Supported platforms are: %s"
                % (self.platform_name, ", ".join(sorted(supported_platforms)))
            )

        try:
            self.platform = openmm.Platform.getPlatformByName(self.platform_name)
        except Exception as exc:
            raise RuntimeError(
                "Requested OpenMM platform '%s' is unavailable" % self.platform_name
            ) from exc
        if self.platform_name == "CUDA":
            self.platform.setPropertyDefaultValue("Precision", "mixed")

    def add_metadynamics(
        self, collective_variables, height=1.0, frequency=500, sigma=None
    ):
        """
        Add metadynamics bias to the simulation.

        Parameters
        ----------
        collective_variables : list
            List of collective variable definitions
        height : float, optional
            Height of Gaussian hills in kJ/mol (default: 1.0)
        frequency : int, optional
            Frequency of hill deposition in steps (default: 500)
        sigma : list, optional
            Widths of Gaussian hills for each CV

        Returns
        -------
        self : Simulation
            Returns self for method chaining
        """
        if sigma is None:
            sigma = [0.1] * len(collective_variables)

        # Create bias variables
        self.bias_variables = []
        for i, (cv_def, s) in enumerate(zip(collective_variables, sigma)):
            if cv_def["type"] == "distance":
                # Distance between two atoms
                atom1, atom2 = cv_def["atoms"]
                bias_var = BiasVariable(
                    openmm.CustomBondForce("r"),
                    minValue=cv_def.get("min", 0.0) * unit.nanometer,
                    maxValue=cv_def.get("max", 2.0) * unit.nanometer,
                    biasWidth=s * unit.nanometer,
                )
                bias_var.addBond([atom1, atom2])
                self.bias_variables.append(bias_var)

        # Create metadynamics object
        self.metadynamics = Metadynamics(
            self.system,
            self.bias_variables,
            self.temperature_quantity,
            biasFactor=10,
            height=height * unit.kilojoules_per_mole,
            frequency=frequency,
        )

        return self

    def minimize_energy(self, max_iterations=1000):
        """
        Minimize the system energy.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of minimization steps (default: 1000)

        Returns
        -------
        self : Simulation
            Returns self for method chaining
        """
        if self.system is None:
            raise RuntimeError("System not prepared. Call prepare_system() first.")

        # Create integrator for minimization
        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature_quantity, 1 / unit.picosecond, 0.002 * unit.picoseconds
        )

        # Create simulation object
        self.simulation = app.Simulation(
            self.pdb.topology, self.system, integrator, self.platform
        )
        self.simulation.context.setPositions(self.pdb.positions)

        # Minimize
        print(f"Minimizing energy for {max_iterations} steps...")
        self.simulation.minimizeEnergy(maxIterations=max_iterations)

        # Get minimized energy
        state = self.simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        print(f"Minimized potential energy: {energy}")

        return self

    def equilibrate(self, steps=10000, report_interval=1000):
        """
        Equilibrate the system.

        Parameters
        ----------
        steps : int, optional
            Number of equilibration steps (default: 10000)
        report_interval : int, optional
            Frequency of progress reports (default: 1000)

        Returns
        -------
        self : Simulation
            Returns self for method chaining
        """
        if self.simulation is None:
            raise RuntimeError("System not minimized. Call minimize_energy() first.")

        print(f"Equilibrating for {steps} steps...")

        # Add reporters for equilibration
        self.simulation.reporters.append(
            app.StateDataReporter(
                f"{self.output_dir}/equilibration.log",
                report_interval,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
            )
        )

        # Run equilibration
        self.simulation.step(steps)

        print("Equilibration complete.")
        return self

    def run_production(
        self,
        steps: int | None = None,
        report_interval: int | None = None,
        *,
        save_trajectory: bool = True,
        bias_hook: Optional[BiasHook] = None,
    ) -> str:
        """Run a short production simulation with deterministic defaults.

        Parameters
        ----------
        steps:
            Total number of integration steps.  If omitted, the value provided
            at construction time is used.
        report_interval:
            Interval (in steps) for data reporters.  Defaults to the minimum of
            ``1000`` and the requested number of steps, but never less than 1.
        save_trajectory:
            Whether to write a DCD trajectory during the run.
        bias_hook:
            Optional callable that injects bias energies per frame.  Reserved for
            advanced workflows.

        Returns
        -------
        str
            Path to the generated trajectory file.  The file is guaranteed to
            exist if ``save_trajectory`` is ``True``.
        """

        if self.openmm_simulation is None:
            raise RuntimeError("System not prepared. Call prepare_system() first.")

        total_steps = int(steps if steps is not None else self.steps)
        if total_steps <= 0:
            raise ValueError("Production steps must be positive")

        stride = (
            report_interval if report_interval is not None else min(1000, total_steps)
        )
        stride = max(1, int(stride))

        simulation = self.openmm_simulation
        simulation.reporters.clear()

        trajectory_path = self.output_dir / "trajectory.dcd"
        final_path = self.output_dir / "final.xml"

        exit_stack = ExitStack()
        new_reporters: list[Any] = []

        try:
            log_handle = exit_stack.enter_context(
                open(self.output_dir / "production.log", "w", encoding="utf-8")
            )
            state_reporter = app.StateDataReporter(
                log_handle,
                stride,
                step=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
            )
            new_reporters.append(state_reporter)
            simulation.reporters.append(state_reporter)

            if save_trajectory:
                dcd_reporter = app.DCDReporter(str(trajectory_path), stride)
                new_reporters.append(dcd_reporter)
                simulation.reporters.append(dcd_reporter)

            self.bias_hook = bias_hook
            simulation.step(total_steps)

            final_state = simulation.context.getState(
                getPositions=True, getVelocities=True, getEnergy=True
            )
            with open(final_path, "w", encoding="utf-8") as fh:
                fh.write(openmm.XmlSerializer.serialize(final_state))
        finally:
            for reporter in new_reporters:
                if reporter in simulation.reporters:
                    simulation.reporters.remove(reporter)
                _close_reporter_file(reporter)
            exit_stack.close()

        return str(trajectory_path if save_trajectory else final_path)

    def production_run(
        self,
        steps=100000,
        report_interval=1000,
        save_trajectory=True,
        bias_hook: Optional[BiasHook] = None,
    ):
        """Backward-compatible wrapper around :meth:`run_production`."""

        self.run_production(
            steps=steps,
            report_interval=report_interval,
            save_trajectory=save_trajectory,
            bias_hook=bias_hook,
        )
        return self

    def feature_extraction(self, feature_specs=None):
        """
        Extract features from the simulation trajectory.

        Parameters
        ----------
        feature_specs : list, optional
            List of feature specifications to extract

        Returns
        -------
        features : dict
            Dictionary of extracted features
        """
        feature_specs = self._resolve_feature_specs(feature_specs)

        trajectory_file = f"{self.output_dir}/trajectory.dcd"
        topology_file = self.pdb_file

        traj = self._load_trajectory(trajectory_file, topology_file)

        features: Dict[str, Any] = {}
        handlers: Dict[
            str, Callable[[md.Trajectory, Dict[str, Any]], Dict[str, Any]]
        ] = {
            "distances": self._extract_distance_features,
            "angles": self._extract_angle_features,
            "dihedrals": self._extract_dihedral_features,
            "ramachandran": self._extract_ramachandran_features,
        }

        for spec in feature_specs:
            feature_type = spec.get("type")
            handler = handlers.get(feature_type)
            if handler is None:
                continue
            result = handler(traj, spec)
            if result:
                features.update(result)

        return features

    def _resolve_feature_specs(self, feature_specs):
        if feature_specs is None:
            return [
                {"type": "distances", "indices": [[0, 1]]},
                {"type": "angles", "indices": [[0, 1, 2]]},
            ]
        return feature_specs

    def _load_trajectory(self, trajectory_file: str, topology_file: str):
        try:
            return md.load(trajectory_file, top=topology_file)
        except Exception as exc:  # pragma: no cover - log path
            raise RuntimeError(
                "Failed to load trajectory for feature extraction "
                f"from '{trajectory_file}' with topology '{topology_file}'."
            ) from exc

    def _extract_distance_features(
        self, traj: md.Trajectory, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        indices = spec.get("indices")
        if not indices:
            return {}
        distances = md.compute_distances(traj, indices)
        return {"distances": distances}

    def _extract_angle_features(
        self, traj: md.Trajectory, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        indices = spec.get("indices")
        if not indices:
            return {}
        angles = md.compute_angles(traj, indices)
        return {"angles": angles}

    def _extract_dihedral_features(
        self, traj: md.Trajectory, spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        indices = spec.get("indices")
        if not indices:
            return {}
        dihedrals = md.compute_dihedrals(traj, indices)
        return {"dihedrals": dihedrals}

    def _extract_ramachandran_features(
        self, traj: md.Trajectory, _spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        _, phi_angles = md.compute_phi(traj)
        _, psi_angles = md.compute_psi(traj)
        if phi_angles.size == 0 and psi_angles.size == 0:
            return {}
        return {"ramachandran": {"phi": phi_angles, "psi": psi_angles}}

    def build_transition_model(self, features, n_states=50, lag_time=1):
        """
        Build a Markov state model from extracted features.

        Parameters
        ----------
        features : dict
            Features extracted from trajectory
        n_states : int, optional
            Number of microstates for MSM (default: 50)
        lag_time : int, optional
            Lag time for MSM construction (default: 1)

        Returns
        -------
        msm_result : dict
            MSM analysis results
        """
        if not features:
            print("Warning: No features available for MSM construction")
            return {}

        try:
            # Use PMARLO's MSM building capabilities
            # Combine all features into a single array
            feature_data = []
            for key, values in features.items():
                if isinstance(values, np.ndarray):
                    if values.ndim == 1:
                        values = values.reshape(-1, 1)
                    feature_data.append(values)

            if not feature_data:
                return {}

            X = np.concatenate(feature_data, axis=1)

            # Build MSM using PMARLO API
            msm_result = api.build_msm(
                X, n_clusters=n_states * 2, n_states=n_states, lag_time=lag_time
            )

            return msm_result

        except Exception as e:
            print(f"Warning: MSM construction failed: {e}")
            return {}

    def relative_energies(self, reference_state=0):
        """
        Calculate relative free energies between states.

        Parameters
        ----------
        reference_state : int, optional
            Index of reference state (default: 0)

        Returns
        -------
        energies : np.ndarray
            Relative free energies in kJ/mol
        """
        # This would typically use the MSM stationary distribution
        # For now, return placeholder
        print("Warning: Relative energy calculation not fully implemented")
        return np.array([0.0])  # Placeholder

    def plot_DG(self, save_path=None):
        """
        Plot free energy landscape.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot

        Returns
        -------
        fig : matplotlib.figure.Figure
            The generated figure
        """
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))

            # Placeholder plot - would normally show FES
            ax.text(
                0.5,
                0.5,
                "Free Energy Landscape\n(Implementation pending)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel("Collective Variable 1")
            ax.set_ylabel("Collective Variable 2")
            ax.set_title("Free Energy Surface")

            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved to {save_path}")

            return fig

        except ImportError:
            print("Warning: matplotlib not available for plotting")
            return None

    def save_checkpoint(self, filename=None):
        """
        Save simulation checkpoint.

        Parameters
        ----------
        filename : str, optional
            Checkpoint filename (default: auto-generated)

        Returns
        -------
        str
            Path to saved checkpoint file
        """
        if filename is None:
            filename = f"{self.output_dir}/checkpoint.xml"

        if self.simulation is None:
            raise RuntimeError("No simulation to checkpoint")

        # Save state
        state = self.simulation.context.getState(
            getPositions=True, getVelocities=True, getForces=True, getEnergy=True
        )

        with open(filename, "w") as f:
            f.write(openmm.XmlSerializer.serialize(state))

        print(f"Checkpoint saved to {filename}")
        return filename

    def load_checkpoint(self, filename):
        """
        Load simulation checkpoint.

        Parameters
        ----------
        filename : str
            Checkpoint filename to load

        Returns
        -------
        self : Simulation
            Returns self for method chaining
        """
        if self.simulation is None:
            raise RuntimeError("Simulation not initialized")

        with open(filename, "r") as f:
            state = openmm.XmlSerializer.load(f.read())

        self.simulation.context.setState(state)
        print(f"Checkpoint loaded from {filename}")
        return self

    def get_summary(self):
        """
        Get simulation summary information.

        Returns
        -------
        dict
            Summary of simulation parameters and results
        """
        summary = {
            "pdb_file": self.pdb_file,
            "output_dir": str(self.output_dir),
            "temperature": self.temperature,
            "pressure": self.pressure,
            "platform": self.platform_name,
            "system_prepared": self.system is not None,
            "simulation_initialized": self.simulation is not None,
            "metadynamics_enabled": self.metadynamics is not None,
            "num_bias_variables": len(self.bias_variables),
        }

        if self.pdb is not None:
            summary["num_atoms"] = self.pdb.topology.getNumAtoms()
            summary["num_residues"] = self.pdb.topology.getNumResidues()

        return summary


# Convenience functions for common workflows
def prepare_system(
    pdb_file,
    forcefield_files=None,
    water_model="tip3p",
    pdb_file_name=None,
):
    """
    Prepare a molecular system for simulation.

    Parameters
    ----------
    pdb_file : str
        Path to PDB file
    forcefield_files : list, optional
        Force field XML files
    water_model : str, optional
        Water model to use
    pdb_file_name : str, optional
        Alternative path to the input PDB file; takes precedence when provided

    Returns
    -------
    Simulation
        Prepared simulation object
    """
    pdb_path = pdb_file_name or pdb_file
    if pdb_path is None:
        raise ValueError("pdb_file or pdb_file_name must be provided")

    sim = Simulation(pdb_path)
    sim.prepare_system(forcefield_files, water_model)
    return sim


def production_run(
    sim, steps=100000, report_interval=1000, bias_hook: Optional[BiasHook] = None
):
    """
    Run a production simulation.

    Parameters
    ----------
    sim : Simulation
        Prepared simulation object
    steps : int, optional
        Number of simulation steps
    report_interval : int, optional
        Reporting frequency
    bias_hook : BiasHook | None, optional
        If provided, bias_hook(cv_values) must return per-frame bias potentials in CV space.

    Returns
    -------
    Simulation
        Simulation object after production run
    """
    return sim.production_run(
        steps=steps, report_interval=report_interval, bias_hook=bias_hook
    )


def feature_extraction(sim, feature_specs=None):
    """
    Extract features from simulation trajectory.

    Parameters
    ----------
    sim : Simulation
        Simulation object with completed trajectory
    feature_specs : list, optional
        Feature specifications

    Returns
    -------
    dict
        Extracted features
    """
    return sim.feature_extraction(feature_specs)


def build_transition_model(features, n_states=50, lag_time=1):
    """
    Build Markov state model from features.

    Parameters
    ----------
    features : dict
        Extracted features
    n_states : int, optional
        Number of states
    lag_time : int, optional
        Lag time for transitions

    Returns
    -------
    dict
        MSM results
    """
    # This is a standalone function that doesn't require a Simulation object
    try:
        feature_data = []
        for key, values in features.items():
            if isinstance(values, np.ndarray):
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                feature_data.append(values)

        if not feature_data:
            return {}

        X = np.concatenate(feature_data, axis=1)
        msm_result = api.build_msm(
            X, n_clusters=n_states * 2, n_states=n_states, lag_time=lag_time
        )
        return msm_result

    except Exception as e:
        print(f"Warning: MSM construction failed: {e}")
        return {}


def relative_energies(msm_result, reference_state=0):
    """
    Calculate relative free energies from MSM.

    Parameters
    ----------
    msm_result : dict
        MSM analysis results
    reference_state : int, optional
        Reference state index

    Returns
    -------
    np.ndarray
        Relative free energies
    """
    # Placeholder implementation
    print("Warning: Relative energy calculation not fully implemented")
    return np.array([0.0])


def plot_DG(features, save_path=None):
    """
    Plot free energy landscape.

    Parameters
    ----------
    features : dict
        Extracted features or MSM results
    save_path : str, optional
        Path to save plot

    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "Free Energy Landscape\n(Implementation pending)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("Collective Variable 1")
        ax.set_ylabel("Collective Variable 2")
        ax.set_title("Free Energy Surface")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    except ImportError:
        print("Warning: matplotlib not available for plotting")
        return None


logger = logging.getLogger(__name__)


def _close_reporter_file(reporter: Any) -> None:
    """Ensure OpenMM reporters release underlying file handles."""
    try:
        opened = getattr(reporter, "_openedFile")
    except AttributeError:
        opened = None
    if opened is False:
        return
    handle = getattr(reporter, "_out", None)
    if handle is None:
        return
    close = getattr(handle, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "Failed to close OpenMM reporter %r cleanly", reporter, exc_info=True
        )
