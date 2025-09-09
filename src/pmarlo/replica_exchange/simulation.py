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
        OpenMM platform to use ("CUDA", "OpenCL", "CPU", "Reference")
    """

    def __init__(
        self,
        pdb_file: str,
        output_dir: str = "output",
        temperature: float = 300.0,
        pressure: float = 1.0,
        platform: str = "CUDA",
    ):
        self.pdb_file = pdb_file
        self.output_dir = output_dir
        self.temperature = temperature * unit.kelvin
        self.pressure = pressure * unit.bar
        self.platform_name = platform

        # Initialize OpenMM objects
        self.pdb = None
        self.forcefield = None
        self.system = None
        self.simulation = None
        self.platform = None

        # Trajectory storage
        self.trajectory_data = []
        self.energies = defaultdict(list)

        # Metadynamics setup
        self.metadynamics = None
        self.bias_variables = []

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
        self : Simulation
            Returns self for method chaining
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
        barostat = openmm.MonteCarloBarostat(self.pressure, self.temperature)
        self.system.addForce(barostat)

        # Set up platform
        self._setup_platform()

        return self

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
        """Set up the OpenMM platform."""
        try:
            self.platform = openmm.Platform.getPlatformByName(self.platform_name)
            if self.platform_name == "CUDA":
                self.platform.setPropertyDefaultValue("Precision", "mixed")
        except Exception:
            # Fall back to CPU if requested platform is not available
            self.platform = openmm.Platform.getPlatformByName("CPU")
            print(f"Warning: {self.platform_name} platform not available, using CPU")

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
            self.temperature,
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
            self.temperature, 1 / unit.picosecond, 0.002 * unit.picoseconds
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

    def production_run(self, steps=100000, report_interval=1000, save_trajectory=True):
        """
        Run production molecular dynamics simulation.

        Parameters
        ----------
        steps : int, optional
            Number of production steps (default: 100000)
        report_interval : int, optional
            Frequency of trajectory and energy reporting (default: 1000)
        save_trajectory : bool, optional
            Whether to save trajectory to file (default: True)

        Returns
        -------
        self : Simulation
            Returns self for method chaining
        """
        if self.simulation is None:
            raise RuntimeError("System not equilibrated. Call equilibrate() first.")

        print(f"Running production simulation for {steps} steps...")

        # Clear previous reporters
        self.simulation.reporters.clear()

        # Add energy reporter
        self.simulation.reporters.append(
            app.StateDataReporter(
                f"{self.output_dir}/production.log",
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

        # Add trajectory reporter if requested
        if save_trajectory:
            self.simulation.reporters.append(
                app.DCDReporter(f"{self.output_dir}/trajectory.dcd", report_interval)
            )

        # Run production
        self.simulation.step(steps)

        print("Production simulation complete.")
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
        if feature_specs is None:
            feature_specs = [
                {"type": "distances", "indices": [[0, 1]]},
                {"type": "angles", "indices": [[0, 1, 2]]},
            ]

        # Load trajectory
        trajectory_file = f"{self.output_dir}/trajectory.dcd"
        topology_file = self.pdb_file

        try:
            traj = md.load(trajectory_file, top=topology_file)
        except Exception as e:
            print(f"Warning: Could not load trajectory: {e}")
            return {}

        features = {}

        for spec in feature_specs:
            if spec["type"] == "distances":
                distances = md.compute_distances(traj, spec["indices"])
                features["distances"] = distances

            elif spec["type"] == "angles":
                angles = md.compute_angles(traj, spec["indices"])
                features["angles"] = angles

            elif spec["type"] == "dihedrals":
                dihedrals = md.compute_dihedrals(traj, spec["indices"])
                features["dihedrals"] = dihedrals

            elif spec["type"] == "ramachandran":
                # Compute phi/psi angles for all residues
                phi_indices, psi_indices = [], []
                for residue in traj.topology.residues:
                    phi_atoms = [
                        atom.index
                        for atom in residue.atoms
                        if atom.name in ["C", "N", "CA", "C"]
                    ]
                    if len(phi_atoms) == 4:
                        phi_indices.append(phi_atoms)

                if phi_indices:
                    phi_angles = md.compute_dihedrals(traj, phi_indices)
                    psi_angles = md.compute_dihedrals(traj, psi_indices)
                    features["ramachandran"] = {"phi": phi_angles, "psi": psi_angles}

        return features

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
            "output_dir": self.output_dir,
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
def prepare_system(pdb_file, forcefield_files=None, water_model="tip3p"):
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

    Returns
    -------
    Simulation
        Prepared simulation object
    """
    sim = Simulation(pdb_file)
    sim.prepare_system(forcefield_files, water_model)
    return sim


def production_run(sim, steps=100000, report_interval=1000):
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

    Returns
    -------
    Simulation
        Simulation object after production run
    """
    return sim.production_run(steps, report_interval)


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
