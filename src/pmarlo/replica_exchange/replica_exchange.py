# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Replica Exchange Molecular Dynamics (REMD) implementation for enhanced sampling.

This module provides functionality to run replica exchange simulations using OpenMM,
allowing for better exploration of conformational space across multiple temperatures.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import openmm
from openmm import Platform, unit
from openmm.app import PME, DCDReporter, ForceField, HBonds, PDBFile, Simulation

from ..utils.replica_utils import exponential_temperature_ladder

logger = logging.getLogger(__name__)


class ReplicaExchange:
    """
    Replica Exchange Molecular Dynamics implementation using OpenMM.

    This class handles the setup and execution of REMD simulations,
    managing multiple temperature replicas and exchange attempts.
    """

    def __init__(
        self,
        pdb_file: str,
        forcefield_files: Optional[List[str]] = None,
        temperatures: Optional[List[float]] = None,
        output_dir: str = "output/replica_exchange",
        exchange_frequency: int = 50,  # Very frequent exchanges for testing
        auto_setup: bool = False,
        dcd_stride: int = 1000,
    ):  # Explicit opt-in for auto-setup
        """
        Initialize the replica exchange simulation.

        Args:
            pdb_file: Path to the prepared PDB file
            forcefield_files: List of forcefield XML files
            temperatures: List of temperatures in Kelvin for replicas
            output_dir: Directory to store output files
            exchange_frequency: Number of steps between exchange attempts
            auto_setup: Whether to automatically set up replicas during initialization
        """
        self.pdb_file = pdb_file
        self.forcefield_files = forcefield_files or [
            "amber14-all.xml",
            "amber14/tip3pfb.xml",
        ]
        self.temperatures = temperatures or self._generate_temperature_ladder()
        self.output_dir = Path(output_dir)
        self.exchange_frequency = exchange_frequency
        self.dcd_stride = dcd_stride

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize replicas - Fixed: Added proper type annotations
        self.n_replicas = len(self.temperatures)
        self.replicas: List[Simulation] = (
            []
        )  # Fixed: Added type annotation for Simulation objects
        self.contexts: List[openmm.Context] = (
            []
        )  # Fixed: Added type annotation for OpenMM Context objects
        self.integrators: List[openmm.Integrator] = (
            []
        )  # Fixed: Added type annotation for OpenMM Integrator objects
        self._is_setup = False  # Track setup state

        # Exchange statistics
        self.exchange_attempts = 0
        self.exchanges_accepted = 0
        self.replica_states = list(
            range(self.n_replicas)
        )  # Which temperature each replica is at
        self.state_replicas = list(
            range(self.n_replicas)
        )  # Which replica is at each temperature
        # Per-pair statistics (temperature index pairs)
        self.pair_attempt_counts: dict[tuple[int, int], int] = {}
        self.pair_accept_counts: dict[tuple[int, int], int] = {}

        # Simulation data - Fixed: Added proper type annotations
        self.trajectory_files: List[Path] = (
            []
        )  # Fixed: Added type annotation for Path objects
        self.energies: List[float] = []  # Fixed: Added type annotation for float values
        self.exchange_history: List[List[int]] = (
            []
        )  # Fixed: Added type annotation for nested int lists

        logger.info(f"Initialized REMD with {self.n_replicas} replicas")
        logger.info(
            (
                f"Temperature range: {min(self.temperatures):.1f} - "
                f"{max(self.temperatures):.1f} K"
            )
        )

        # Auto-setup if requested (for API consistency)
        if auto_setup:
            logger.info("Auto-setting up replicas...")
            self.setup_replicas()

    def _generate_temperature_ladder(
        self,
        min_temp: float = 300.0,
        max_temp: float = 350.0,
        n_replicas: int = 3,
    ) -> List[float]:
        """
        Generate an exponential temperature ladder for optimal exchange
        efficiency.

        Delegates to
        `utils.replica_utils.exponential_temperature_ladder` to avoid
        duplication.
        """
        return exponential_temperature_ladder(min_temp, max_temp, n_replicas)

    def setup_replicas(self, bias_variables: Optional[List] = None):
        """
        Set up all replica simulations with different temperatures.

        Args:
            bias_variables: Optional list of bias variables for metadynamics
        """
        logger.info("Setting up replica simulations...")

        pdb, forcefield = self._load_pdb_and_forcefield()
        system = self._create_system(pdb, forcefield)
        self._log_system_info(system)
        self._setup_metadynamics(system, bias_variables)
        platform, platform_properties = self._select_platform_and_properties()

        shared_minimized_positions = None

        for i, temperature in enumerate(self.temperatures):
            logger.info(f"Setting up replica {i} at {temperature}K...")

            integrator = self._create_integrator_for_temperature(temperature)
            simulation = self._create_simulation(
                pdb, system, integrator, platform, platform_properties
            )
            simulation.context.setPositions(pdb.positions)

            if (
                shared_minimized_positions is not None
                and self._reuse_minimized_positions_quick_minimize(
                    simulation, shared_minimized_positions, i
                )
            ):
                traj_file = self._add_dcd_reporter(simulation, i)
                self._store_replica_data(simulation, integrator, traj_file)
                logger.info(f"Replica {i:02d}: T = {temperature:.1f} K")
                continue

            logger.info(f"  Minimizing energy for replica {i}...")
            self._check_initial_energy(simulation, i)
            minimization_success = self._perform_stage1_minimization(simulation, i)

            if minimization_success:
                shared_minimized_positions = (
                    self._perform_stage2_minimization_and_validation(
                        simulation, i, shared_minimized_positions
                    )
                )

            traj_file = self._add_dcd_reporter(simulation, i)
            self._store_replica_data(simulation, integrator, traj_file)
            logger.info(f"Replica {i:02d}: T = {temperature:.1f} K")

        logger.info("All replicas set up successfully")
        self._is_setup = True

    # --- Helper methods for setup_replicas ---

    def _load_pdb_and_forcefield(self) -> Tuple[PDBFile, ForceField]:
        pdb = PDBFile(self.pdb_file)
        forcefield = ForceField(*self.forcefield_files)
        return pdb, forcefield

    def _create_system(self, pdb: PDBFile, forcefield: ForceField) -> openmm.System:
        logger.info("Creating molecular system with HMR and tuned parameters...")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            constraints=HBonds,
            rigidWater=True,
            nonbondedCutoff=0.9 * unit.nanometer,
            ewaldErrorTolerance=1e-4,
            hydrogenMass=3.0 * unit.amu,
            removeCMMotion=True,
        )
        return system

    def _log_system_info(self, system: openmm.System) -> None:
        logger.info(f"System created with {system.getNumParticles()} particles")
        logger.info(f"System has {system.getNumForces()} force terms")
        for force_idx in range(system.getNumForces()):
            force = system.getForce(force_idx)
            logger.info(f"  Force {force_idx}: {force.__class__.__name__}")

    def _setup_metadynamics(
        self, system: openmm.System, bias_variables: Optional[List]
    ) -> None:
        self.metadynamics = None
        if not bias_variables:
            return
        from openmm.app.metadynamics import Metadynamics

        bias_dir = self.output_dir / "bias"
        bias_dir.mkdir(exist_ok=True)
        self.metadynamics = Metadynamics(
            system,
            bias_variables,
            temperature=self.temperatures[0] * unit.kelvin,
            biasFactor=10.0,
            height=1.0 * unit.kilojoules_per_mole,
            frequency=500,
            biasDir=str(bias_dir),
            saveFrequency=1000,
        )

    def _select_platform_and_properties(
        self,
    ) -> Tuple[Platform, Dict[str, str]]:
        platform_properties: Dict[str, str] = {}
        try:
            platform = Platform.getPlatformByName("CUDA")
            platform_properties = {
                "Precision": "mixed",
                "UseFastMath": "true",
                "DeterministicForces": "false",
            }
            logger.info("Using CUDA (mixed precision, fast math)")
        except Exception:
            try:
                try:
                    platform = Platform.getPlatformByName("HIP")
                    logger.info("Using HIP (AMD GPU)")
                except Exception:
                    platform = Platform.getPlatformByName("OpenCL")
                    logger.info("Using OpenCL")
            except Exception:
                platform = Platform.getPlatformByName("CPU")
                try:
                    Platform.setPropertyDefaultValue(
                        "CpuThreads", str(os.cpu_count() or 1)
                    )
                except Exception:
                    pass
                logger.info("Using CPU with all cores")
        return platform, platform_properties

    def _create_integrator_for_temperature(
        self, temperature: float
    ) -> openmm.Integrator:
        return openmm.LangevinIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picosecond,
            2.0 * unit.femtoseconds,
        )

    def _create_simulation(
        self,
        pdb: PDBFile,
        system: openmm.System,
        integrator: openmm.Integrator,
        platform: Platform,
        platform_properties: Dict[str, str],
    ) -> Simulation:
        return Simulation(
            pdb.topology, system, integrator, platform, platform_properties or None
        )

    def _reuse_minimized_positions_quick_minimize(
        self,
        simulation: Simulation,
        shared_minimized_positions,
        replica_index: int,
    ) -> bool:
        try:
            simulation.context.setPositions(shared_minimized_positions)
            simulation.minimizeEnergy(
                maxIterations=50,
                tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
            )
            logger.info(
                (
                    f"  Reused minimized coordinates for replica {replica_index} "
                    f"(quick touch-up)"
                )
            )
            return True
        except Exception as exc:
            logger.warning(
                (
                    f"  Failed to reuse minimized coords for replica "
                    f"{replica_index}: {exc}; falling back to full minimization"
                )
            )
            return False

    def _check_initial_energy(self, simulation: Simulation, replica_index: int) -> None:
        try:
            initial_state = simulation.context.getState(getEnergy=True)
            initial_energy = initial_state.getPotentialEnergy()
            logger.info(
                f"  Initial energy for replica {replica_index}: {initial_energy}"
            )
            energy_val = initial_energy.value_in_unit(unit.kilojoules_per_mole)
            if abs(energy_val) > 1e6:
                logger.warning(
                    (
                        "  Very high initial energy ("
                        f"{energy_val:.2e} kJ/mol) detected for replica "
                        f"{replica_index}"
                    )
                )
        except Exception as exc:
            logger.warning(
                f"  Could not check initial energy for replica {replica_index}: {exc}"
            )

    def _perform_stage1_minimization(
        self, simulation: Simulation, replica_index: int
    ) -> bool:
        minimization_success = False
        schedule = [(50, 100.0), (100, 50.0), (200, 10.0)]
        for attempt, (max_iter, tolerance_val) in enumerate(schedule):
            try:
                tolerance = tolerance_val * unit.kilojoules_per_mole / unit.nanometer
                simulation.minimizeEnergy(maxIterations=max_iter, tolerance=tolerance)
                logger.info(
                    (
                        "  Stage 1 minimization completed for replica "
                        f"{replica_index} (attempt {attempt + 1})"
                    )
                )
                minimization_success = True
                break
            except Exception as exc:
                logger.warning(
                    (
                        "  Stage 1 minimization attempt "
                        f"{attempt + 1} failed for replica {replica_index}: {exc}"
                    )
                )
                if attempt == len(schedule) - 1:
                    logger.error(
                        f"  All minimization attempts failed for replica {replica_index}"
                    )
                    raise RuntimeError(
                        (
                            f"Energy minimization failed for replica {replica_index} "
                            "after 3 attempts. Structure may be too distorted. "
                            "Consider: 1) Better initial structure, 2) Different "
                            "forcefield, 3) Manual structure preparation"
                        )
                    )
        return minimization_success

    def _perform_stage2_minimization_and_validation(
        self,
        simulation: Simulation,
        replica_index: int,
        shared_minimized_positions,
    ):
        try:
            self._stage2_minimize(simulation, replica_index)
            state = self._get_state_with_positions(simulation)
            energy = state.getPotentialEnergy()
            positions = state.getPositions()
            self._validate_energy(energy, replica_index)
            self._validate_positions(positions, replica_index)
            logger.info(f"  Final energy for replica {replica_index}: {energy}")
            if shared_minimized_positions is None:
                shared_minimized_positions = self._cache_minimized_positions_safe(state)
            return shared_minimized_positions
        except Exception as exc:
            self._log_stage2_failure(replica_index, exc)
            self._log_using_stage1_energy(simulation, replica_index)
            return shared_minimized_positions

    # ---- Helpers for stage 2 minimization (split for C901) ----

    def _stage2_minimize(self, simulation: Simulation, replica_index: int) -> None:
        simulation.minimizeEnergy(
            maxIterations=100, tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer
        )
        logger.info(f"  Stage 2 minimization completed for replica {replica_index}")

    def _get_state_with_positions(self, simulation: Simulation):
        return simulation.context.getState(
            getPositions=True, getEnergy=True, getVelocities=True
        )

    def _validate_energy(self, energy, replica_index: int) -> None:
        energy_str = str(energy).lower()
        if "nan" in energy_str or "inf" in energy_str:
            raise ValueError(
                (
                    "Invalid energy ("
                    f"{energy}) detected after minimization for replica "
                    f"{replica_index}"
                )
            )
        energy_val = energy.value_in_unit(unit.kilojoules_per_mole)
        if abs(energy_val) > 1e5:
            logger.warning(
                (
                    f"  High final energy ({energy_val:.2e} kJ/mol) for "
                    f"replica {replica_index}"
                )
            )

    def _validate_positions(self, positions, replica_index: int) -> None:
        pos_array = positions.value_in_unit(unit.nanometer)
        if np.any(np.isnan(pos_array)) or np.any(np.isinf(pos_array)):
            raise ValueError(
                (
                    "Invalid positions detected after minimization for "
                    f"replica {replica_index}"
                )
            )

    def _cache_minimized_positions_safe(self, state):
        try:
            logger.info("  Cached minimized coordinates from replica 0 for reuse")
            return state.getPositions()
        except Exception:
            return None

    def _log_stage2_failure(self, replica_index: int, exc: Exception) -> None:
        logger.error(
            (
                "  Stage 2 minimization or validation failed for replica "
                f"{replica_index}: {exc}"
            )
        )
        logger.warning(
            (
                "  Attempting to continue with Stage 1 result for replica "
                f"{replica_index}"
            )
        )

    def _log_using_stage1_energy(
        self, simulation: Simulation, replica_index: int
    ) -> None:
        try:
            state = simulation.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy()
            logger.info(f"  Using Stage 1 energy for replica {replica_index}: {energy}")
        except Exception:
            raise RuntimeError(
                f"Complete minimization failure for replica {replica_index}"
            )

    def _add_dcd_reporter(self, simulation: Simulation, replica_index: int) -> Path:
        traj_file = self.output_dir / f"replica_{replica_index:02d}.dcd"
        dcd_reporter = DCDReporter(str(traj_file), int(max(1, self.dcd_stride)))
        simulation.reporters.append(dcd_reporter)
        return traj_file

    def _store_replica_data(
        self,
        simulation: Simulation,
        integrator: openmm.Integrator,
        traj_file: Path,
    ) -> None:
        self.replicas.append(simulation)
        self.integrators.append(integrator)
        self.contexts.append(simulation.context)
        self.trajectory_files.append(traj_file)

    def is_setup(self) -> bool:
        """
        Check if replicas are properly set up.

        Returns:
            True if replicas are set up, False otherwise
        """
        return (
            self._is_setup
            and len(self.contexts) == self.n_replicas
            and len(self.replicas) == self.n_replicas
        )

    def auto_setup_if_needed(self, bias_variables: Optional[List] = None):
        """
        Automatically set up replicas if not already done.

        Args:
            bias_variables: Optional list of bias variables for metadynamics
        """
        if not self.is_setup():
            logger.info("Auto-setting up replicas...")
            self.setup_replicas(bias_variables=bias_variables)

    def save_checkpoint_state(self) -> Dict[str, Any]:
        """
        Save the current state for checkpointing.

        Returns:
            Dictionary containing the current state
        """
        if not self.is_setup():
            return {"setup": False}

        # Save critical state information
        state = {
            "setup": True,
            "n_replicas": self.n_replicas,
            "temperatures": self.temperatures,
            "replica_states": self.replica_states.copy(),
            "state_replicas": self.state_replicas.copy(),
            "exchange_attempts": self.exchange_attempts,
            "exchanges_accepted": self.exchanges_accepted,
            "exchange_history": self.exchange_history.copy(),
            "output_dir": str(self.output_dir),
            "exchange_frequency": self.exchange_frequency,
        }

        # Save positions and velocities for each replica
        replica_data = []
        for i, context in enumerate(self.contexts):
            try:
                sim_state = context.getState(
                    getPositions=True, getVelocities=True, getEnergy=True
                )
                replica_data.append(
                    {
                        "positions": sim_state.getPositions(),
                        "velocities": sim_state.getVelocities(),
                        "energy": sim_state.getPotentialEnergy(),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not save state for replica {i}: {e}")
                replica_data.append(
                    {  # Fixed: Replace None with empty dict to satisfy type checker
                        "positions": None,
                        "velocities": None,
                        "energy": None,
                    }
                )

        state["replica_data"] = replica_data
        return state

    def restore_from_checkpoint(
        self, checkpoint_state: Dict[str, Any], bias_variables: Optional[List] = None
    ):
        """
        Restore the replica exchange from a checkpoint state.

        Args:
            checkpoint_state: Previously saved state dictionary
            bias_variables: Optional list of bias variables for metadynamics
        """
        if not checkpoint_state.get("setup", False):
            logger.info(
                "Checkpoint indicates replicas were not set up, setting up now..."
            )
            self.setup_replicas(bias_variables=bias_variables)
            return

        logger.info("Restoring replica exchange from checkpoint...")

        # Restore basic state
        self.exchange_attempts = checkpoint_state.get("exchange_attempts", 0)
        self.exchanges_accepted = checkpoint_state.get("exchanges_accepted", 0)
        self.exchange_history = checkpoint_state.get("exchange_history", [])
        self.replica_states = checkpoint_state.get(
            "replica_states", list(range(self.n_replicas))
        )
        self.state_replicas = checkpoint_state.get(
            "state_replicas", list(range(self.n_replicas))
        )

        # If replicas aren't set up, set them up first
        if not self.is_setup():
            logger.info("Setting up replicas for checkpoint restoration...")
            self.setup_replicas(bias_variables=bias_variables)

        # Restore replica states if available
        replica_data = checkpoint_state.get("replica_data", [])
        if replica_data and len(replica_data) == self.n_replicas:
            logger.info("Restoring individual replica states...")
            for i, (context, data) in enumerate(zip(self.contexts, replica_data)):
                if (
                    data is not None and data.get("positions") is not None
                ):  # Fixed: Check for valid data
                    try:
                        # Create a state with the saved positions and velocities
                        context.setPositions(data["positions"])
                        context.setVelocities(data["velocities"])
                        logger.info(f"Restored state for replica {i}")
                    except Exception as e:
                        logger.warning(f"Could not restore state for replica {i}: {e}")
                        # Continue with default state

        logger.info(
            "Checkpoint restoration complete. Exchange stats: "
            f"{self.exchanges_accepted}/{self.exchange_attempts}"
        )

    def calculate_exchange_probability(self, replica_i: int, replica_j: int) -> float:
        """
        Calculate the probability of exchanging two replicas.

        Args:
            replica_i: Index of first replica
            replica_j: Index of second replica

        Returns:
            Exchange probability
        """
        # BOUNDS CHECKING: Ensure replica indices are valid
        if replica_i < 0 or replica_i >= len(self.contexts):
            raise ValueError(
                f"replica_i={replica_i} is out of bounds [0, {len(self.contexts)})"
            )
        if replica_j < 0 or replica_j >= len(self.contexts):
            raise ValueError(
                f"replica_j={replica_j} is out of bounds [0, {len(self.contexts)})"
            )

        # Get current energies
        state_i = self.contexts[replica_i].getState(getEnergy=True)
        state_j = self.contexts[replica_j].getState(getEnergy=True)

        energy_i = state_i.getPotentialEnergy()
        energy_j = state_j.getPotentialEnergy()

        # Get temperatures
        temp_i = self.temperatures[self.replica_states[replica_i]]
        temp_j = self.temperatures[self.replica_states[replica_j]]

        # Calculate exchange probability using canonical Metropolis criterion
        # delta = (beta_j - beta_i) * (U_i - U_j)
        # where beta = 1 / (R T)
        def safe_dimensionless(q):
            if hasattr(q, "value_in_unit"):
                return q.value_in_unit(unit.dimensionless)
            return float(q)

        beta_i = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temp_i * unit.kelvin)
        beta_j = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temp_j * unit.kelvin)
        delta = safe_dimensionless((beta_j - beta_i) * (energy_i - energy_j))
        prob = min(1.0, np.exp(-delta))

        # Debug logging for troubleshooting low acceptance rates
        logger.debug(
            (
                f"Exchange calculation: E_i={energy_i}, E_j={energy_j}, "
                f"T_i={temp_i:.1f}K, T_j={temp_j:.1f}K, "
                f"delta={delta:.3f}, prob={prob:.6f}"
            )
        )

        return float(prob)  # Fixed: Explicit float conversion to avoid Any return type

    def attempt_exchange(
        self,
        replica_i: int,
        replica_j: int,
        energies: Optional[List[openmm.unit.quantity.Quantity]] = None,
    ) -> bool:
        """
        Attempt to exchange two replicas.

        Args:
            replica_i: Index of first replica
            replica_j: Index of second replica

        Returns:
            True if exchange was accepted, False otherwise
        """
        # BOUNDS CHECKING: Ensure replica indices are valid
        self._validate_replica_indices(replica_i, replica_j)

        self.exchange_attempts += 1

        # Calculate exchange probability (use cached energies if provided)
        prob = (
            self._calculate_probability_from_cached(replica_i, replica_j, energies)
            if energies is not None
            else self.calculate_exchange_probability(replica_i, replica_j)
        )

        # Track per-pair stats and perform the exchange if accepted
        state_i_val = self.replica_states[replica_i]
        state_j_val = self.replica_states[replica_j]
        pair = (min(state_i_val, state_j_val), max(state_i_val, state_j_val))
        self.pair_attempt_counts[pair] = self.pair_attempt_counts.get(pair, 0) + 1

        if np.random.random() < prob:
            self._perform_exchange(replica_i, replica_j)
            self.exchanges_accepted += 1
            self.pair_accept_counts[pair] = self.pair_accept_counts.get(pair, 0) + 1
            logger.debug(
                (
                    f"Exchange accepted: replica {replica_i} <-> {replica_j} "
                    f"(prob={prob:.3f})"
                )
            )
            return True

        logger.debug(
            (
                f"Exchange rejected: replica {replica_i} <-> {replica_j} "
                f"(prob={prob:.3f})"
            )
        )
        return False

    # --- Helper methods for attempt_exchange ---

    def _validate_replica_indices(self, replica_i: int, replica_j: int) -> None:
        if replica_i < 0 or replica_i >= self.n_replicas:
            raise ValueError(
                f"replica_i={replica_i} is out of bounds [0, {self.n_replicas})"
            )
        if replica_j < 0 or replica_j >= self.n_replicas:
            raise ValueError(
                f"replica_j={replica_j} is out of bounds [0, {self.n_replicas})"
            )
        if replica_i >= len(self.contexts):
            raise RuntimeError(
                f"replica_i={replica_i} >= len(contexts)={len(self.contexts)}"
            )
        if replica_j >= len(self.contexts):
            raise RuntimeError(
                f"replica_j={replica_j} >= len(contexts)={len(self.contexts)}"
            )

    def _calculate_probability_from_cached(
        self,
        replica_i: int,
        replica_j: int,
        energies: List[openmm.unit.quantity.Quantity],
    ) -> float:
        temp_i = self.temperatures[self.replica_states[replica_i]]
        temp_j = self.temperatures[self.replica_states[replica_j]]

        beta_i = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temp_i * unit.kelvin)
        beta_j = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temp_j * unit.kelvin)

        e_i = energies[replica_i]
        e_j = energies[replica_j]
        if not hasattr(e_i, "value_in_unit"):
            e_i = float(e_i) * unit.kilojoules_per_mole
        if not hasattr(e_j, "value_in_unit"):
            e_j = float(e_j) * unit.kilojoules_per_mole

        delta_q = (beta_j - beta_i) * (e_i - e_j)
        try:
            delta = delta_q.value_in_unit(unit.dimensionless)
        except Exception:
            delta = float(delta_q)
        return float(min(1.0, np.exp(-delta)))

    def _perform_exchange(self, replica_i: int, replica_j: int) -> None:
        if replica_i >= len(self.replica_states):
            raise RuntimeError(
                (
                    "replica_states array too small: "
                    f"{len(self.replica_states)}, need {replica_i + 1}"
                )
            )
        if replica_j >= len(self.replica_states):
            raise RuntimeError(
                (
                    "replica_states array too small: "
                    f"{len(self.replica_states)}, need {replica_j + 1}"
                )
            )

        old_state_i = self.replica_states[replica_i]
        old_state_j = self.replica_states[replica_j]
        if old_state_i >= len(self.state_replicas) or old_state_j >= len(
            self.state_replicas
        ):
            raise RuntimeError(
                (
                    "Invalid state indices: "
                    f"{old_state_i}, {old_state_j} vs array size "
                    f"{len(self.state_replicas)}"
                )
            )

        self.replica_states[replica_i] = old_state_j
        self.replica_states[replica_j] = old_state_i

        self.state_replicas[old_state_i] = replica_j
        self.state_replicas[old_state_j] = replica_i

        self.integrators[replica_i].setTemperature(
            self.temperatures[old_state_j] * unit.kelvin
        )
        self.integrators[replica_j].setTemperature(
            self.temperatures[old_state_i] * unit.kelvin
        )

        self.contexts[replica_i].setVelocitiesToTemperature(
            self.temperatures[old_state_j] * unit.kelvin
        )
        self.contexts[replica_j].setVelocitiesToTemperature(
            self.temperatures[old_state_i] * unit.kelvin
        )

    def run_simulation(
        self,
        total_steps: int = 1000,  # Very fast for testing
        equilibration_steps: int = 100,  # Minimal equilibration
        save_state_frequency: int = 1000,
        checkpoint_manager=None,
    ):
        """
        Run the replica exchange simulation.

        Args:
            total_steps: Total number of MD steps to run
            equilibration_steps: Number of equilibration steps before data collection
            save_state_frequency: Frequency to save simulation states
            checkpoint_manager: CheckpointManager instance for state tracking
        """
        self._validate_setup_state()
        self._log_run_start(total_steps)
        if equilibration_steps > 0:
            self._run_equilibration_phase(equilibration_steps, checkpoint_manager)
        if self._skip_production_if_completed(checkpoint_manager):
            return
        self._mark_production_started(checkpoint_manager)
        self._run_production_phase(
            total_steps, equilibration_steps, save_state_frequency, checkpoint_manager
        )
        self._mark_production_completed(
            total_steps, equilibration_steps, checkpoint_manager
        )
        self._close_dcd_files()
        self._log_final_stats()
        self.save_results()

    # --- Helpers for run_simulation ---

    def _validate_setup_state(self) -> None:
        if not self._is_setup:
            raise RuntimeError(
                "Replicas not properly initialized! Call setup_replicas() first."
            )
        if not self.contexts or len(self.contexts) != self.n_replicas:
            raise RuntimeError(
                (
                    "Replicas not properly initialized! Expected "
                    f"{self.n_replicas} contexts, but got {len(self.contexts)}. "
                    "setup_replicas() may have failed."
                )
            )
        if not self.replicas or len(self.replicas) != self.n_replicas:
            raise RuntimeError(
                (
                    "Replicas not properly initialized! Expected "
                    f"{self.n_replicas} replicas, but got {len(self.replicas)}. "
                    "setup_replicas() may have failed."
                )
            )

    def _log_run_start(self, total_steps: int) -> None:
        logger.info(f"Starting REMD simulation: {total_steps} steps")
        logger.info(f"Exchange attempts every {self.exchange_frequency} steps")

    def _run_equilibration_phase(
        self, equilibration_steps: int, checkpoint_manager
    ) -> None:
        if checkpoint_manager and checkpoint_manager.is_step_completed(
            "gradual_heating"
        ):
            logger.info("Gradual heating already completed ✓")
        else:
            self._run_gradual_heating(equilibration_steps, checkpoint_manager)

        if checkpoint_manager and checkpoint_manager.is_step_completed("equilibration"):
            logger.info("Temperature equilibration already completed ✓")
        else:
            self._run_temperature_equilibration(equilibration_steps, checkpoint_manager)

    def _run_gradual_heating(
        self, equilibration_steps: int, checkpoint_manager
    ) -> None:
        if checkpoint_manager:
            checkpoint_manager.mark_step_started("gradual_heating")
        logger.info(f"Equilibration with gradual heating: {equilibration_steps} steps")
        heating_steps = max(100, equilibration_steps * 40 // 100)
        logger.info(f"   Phase 1: Gradual heating over {heating_steps} steps")
        heating_chunk_size = max(10, heating_steps // 20)
        for heat_step in range(0, heating_steps, heating_chunk_size):
            current_steps = min(heating_chunk_size, heating_steps - heat_step)
            progress_fraction = (heat_step + current_steps) / heating_steps
            for replica_idx, replica in enumerate(self.replicas):
                target_temp = self.temperatures[self.replica_states[replica_idx]]
                current_temp = 50.0 + (target_temp - 50.0) * progress_fraction
                replica.integrator.setTemperature(current_temp * unit.kelvin)
                self._step_with_recovery(
                    replica, current_steps, replica_idx, current_temp
                )
            progress = min(40, (heat_step + current_steps) * 40 // heating_steps)
            temps_preview = [
                50.0
                + (self.temperatures[self.replica_states[i]] - 50.0) * progress_fraction
                for i in range(len(self.replicas))
            ]
            logger.info(
                f"   Heating Progress: {progress}% - Current temps: {temps_preview}"
            )
        if checkpoint_manager:
            checkpoint_manager.mark_step_completed(
                "gradual_heating",
                {
                    "heating_steps": heating_steps,
                    "final_temperatures": [
                        self.temperatures[state] for state in self.replica_states
                    ],
                },
            )

    def _step_with_recovery(
        self, replica: Simulation, steps: int, replica_idx: int, temp_k: float
    ) -> None:
        try:
            replica.step(steps)
        except Exception as exc:
            if "nan" in str(exc).lower():
                logger.warning(
                    (
                        f"   NaN detected in replica {replica_idx} during heating, "
                        "attempting recovery..."
                    )
                )
                replica.context.setVelocitiesToTemperature(temp_k * unit.kelvin)
                small_steps = max(1, steps // 5)
                for recovery_attempt in range(5):
                    try:
                        replica.step(small_steps)
                        break
                    except Exception:
                        if recovery_attempt == 4:
                            raise RuntimeError(
                                f"Failed to recover from NaN in replica {replica_idx}"
                            )
                        replica.context.setVelocitiesToTemperature(
                            temp_k * unit.kelvin * 0.9
                        )
            else:
                raise

    def _run_temperature_equilibration(
        self, equilibration_steps: int, checkpoint_manager
    ) -> None:
        if checkpoint_manager:
            checkpoint_manager.mark_step_started("equilibration")
        temp_equil_steps = max(100, equilibration_steps * 60 // 100)
        logger.info(
            (
                "   Phase 2: Temperature equilibration at target temperatures over "
                f"{temp_equil_steps} steps"
            )
        )
        for replica_idx, replica in enumerate(self.replicas):
            target_temp = self.temperatures[self.replica_states[replica_idx]]
            replica.integrator.setTemperature(target_temp * unit.kelvin)
            replica.context.setVelocitiesToTemperature(target_temp * unit.kelvin)
        equil_chunk_size = max(1, temp_equil_steps // 10)
        for i in range(0, temp_equil_steps, equil_chunk_size):
            current_steps = min(equil_chunk_size, temp_equil_steps - i)
            for replica_idx, replica in enumerate(self.replicas):
                try:
                    replica.step(current_steps)
                except Exception as exc:
                    if "nan" in str(exc).lower():
                        logger.error(
                            (
                                f"   NaN detected in replica {replica_idx} during "
                                "equilibration - simulation unstable"
                            )
                        )
                        if checkpoint_manager:
                            checkpoint_manager.mark_step_failed(
                                "equilibration", str(exc)
                            )
                        raise RuntimeError(
                            (
                                "Simulation became unstable for replica "
                                f"{replica_idx}. Try: 1) Better initial structure, "
                                "2) Smaller timestep, 3) More minimization"
                            )
                        )
                    else:
                        raise
            progress = min(100, 40 + (i + current_steps) * 60 // temp_equil_steps)
            logger.info(
                (
                    f"   Equilibration Progress: {progress}% "
                    f"({equilibration_steps - temp_equil_steps + i + current_steps}/"
                    f"{equilibration_steps} steps)"
                )
            )
        if checkpoint_manager:
            checkpoint_manager.mark_step_completed(
                "equilibration",
                {
                    "equilibration_steps": temp_equil_steps,
                    "total_equilibration": equilibration_steps,
                },
            )
        logger.info("   Equilibration Complete ✓")

    def _skip_production_if_completed(self, checkpoint_manager) -> bool:
        if checkpoint_manager and checkpoint_manager.is_step_completed(
            "production_simulation"
        ):
            logger.info("Production simulation already completed ✓")
            return True
        return False

    def _mark_production_started(self, checkpoint_manager) -> None:
        if checkpoint_manager:
            checkpoint_manager.mark_step_started("production_simulation")

    def _run_production_phase(
        self,
        total_steps: int,
        equilibration_steps: int,
        save_state_frequency: int,
        checkpoint_manager,
    ) -> None:
        production_steps = total_steps - equilibration_steps
        exchange_steps = production_steps // self.exchange_frequency
        logger.info(
            (
                f"Production: {production_steps} steps with "
                f"{exchange_steps} exchange attempts"
            )
        )
        for step in range(exchange_steps):
            self._production_step_all_replicas(step, checkpoint_manager)
            energies = self._precompute_energies()
            self._attempt_all_exchanges(energies)
            self.exchange_history.append(self.replica_states.copy())
            self._log_production_progress(
                step, exchange_steps, total_steps, equilibration_steps
            )
            if (step + 1) * self.exchange_frequency % save_state_frequency == 0:
                self.save_checkpoint(step + 1)

    def _production_step_all_replicas(self, step: int, checkpoint_manager) -> None:
        for replica_idx, replica in enumerate(self.replicas):
            try:
                replica.step(self.exchange_frequency)
            except Exception as exc:
                if "nan" in str(exc).lower():
                    logger.error(
                        "NaN detected in replica %d during production phase",
                        replica_idx,
                    )
                    try:
                        _ = replica.context.getState(
                            getPositions=True, getVelocities=True
                        )
                        logger.info(
                            "Attempting to save current state before failure..."
                        )
                    except Exception:
                        pass
                    raise RuntimeError(
                        (
                            "Simulation became unstable for replica "
                            f"{replica_idx} at production step {step}. "
                            "Consider: 1) Longer equilibration, 2) Smaller timestep, "
                            "3) Different initial structure"
                        )
                    )
                else:
                    raise

    def _precompute_energies(self) -> List[Any]:
        energies: List[Any] = []
        for idx, ctx in enumerate(self.contexts):
            try:
                e_state = ctx.getState(getEnergy=True)
                energies.append(e_state.getPotentialEnergy())
            except Exception as exc:
                logger.debug(f"Energy getState failed for replica {idx}: {exc}")
                last = self.energies[idx] if idx < len(self.energies) else 0.0
                energies.append(last)
        self.energies = energies
        return energies

    def _attempt_all_exchanges(self, energies: List[Any]) -> None:
        for i in range(0, self.n_replicas - 1, 2):
            try:
                self.attempt_exchange(i, i + 1, energies=energies)
            except Exception as exc:
                logger.warning(
                    (
                        f"Exchange attempt failed between replicas {i} and {i+1}: "
                        f"{exc}"
                    )
                )
        for i in range(1, self.n_replicas - 1, 2):
            try:
                self.attempt_exchange(i, i + 1, energies=energies)
            except Exception as exc:
                logger.warning(
                    (
                        f"Exchange attempt failed between replicas {i} and {i+1}: "
                        f"{exc}"
                    )
                )

    def _log_production_progress(
        self, step: int, exchange_steps: int, total_steps: int, equilibration_steps: int
    ) -> None:
        progress_percent = (step + 1) * 100 // exchange_steps
        acceptance_rate = self.exchanges_accepted / max(1, self.exchange_attempts)
        completed_steps = (step + 1) * self.exchange_frequency + equilibration_steps
        logger.debug(
            (
                f"   Production Progress: {progress_percent}% "
                f"({step + 1}/{exchange_steps} exchanges, "
                f"{completed_steps}/{total_steps} total steps) "
                f"| Acceptance: {acceptance_rate:.3f}"
            )
        )

    def _mark_production_completed(
        self, total_steps: int, equilibration_steps: int, checkpoint_manager
    ) -> None:
        production_steps = total_steps - equilibration_steps
        exchange_steps = production_steps // self.exchange_frequency
        if checkpoint_manager:
            checkpoint_manager.mark_step_completed(
                "production_simulation",
                {
                    "production_steps": production_steps,
                    "exchange_steps": exchange_steps,
                    "final_acceptance_rate": self.exchanges_accepted
                    / max(1, self.exchange_attempts),
                },
            )

    def _log_final_stats(self) -> None:
        final_acceptance = self.exchanges_accepted / max(1, self.exchange_attempts)
        logger.info("=" * 60)
        logger.info("🎉 REPLICA EXCHANGE SIMULATION COMPLETED! 🎉")
        logger.info(f"Final exchange acceptance rate: {final_acceptance:.3f}")
        logger.info(f"Total exchanges attempted: {self.exchange_attempts}")
        logger.info(f"Total exchanges accepted: {self.exchanges_accepted}")
        logger.info("=" * 60)

    def _close_dcd_files(self):
        """Close and flush all DCD files to ensure data is written."""
        logger.info("Closing DCD files...")

        for i, replica in enumerate(self.replicas):
            # Remove DCD reporters to force file closure
            dcd_reporters = [r for r in replica.reporters if hasattr(r, "_out")]
            for reporter in dcd_reporters:
                try:
                    # Force close the DCD file
                    if hasattr(reporter, "_out") and reporter._out:
                        reporter._out.close()
                        logger.debug(f"Closed DCD file for replica {i}")
                except Exception as e:
                    logger.warning(f"Error closing DCD file for replica {i}: {e}")

            # Remove DCD reporters from the simulation
            replica.reporters = [r for r in replica.reporters if not hasattr(r, "_out")]

        # Force garbage collection to ensure file handles are released
        import gc

        gc.collect()

        logger.info("DCD files closed and flushed")

    def save_checkpoint(self, step: int):
        """Save simulation checkpoint."""
        checkpoint_file = self.output_dir / f"checkpoint_step_{step:06d}.pkl"
        checkpoint_data = {
            "step": step,
            "replica_states": self.replica_states,
            "state_replicas": self.state_replicas,
            "exchange_attempts": self.exchange_attempts,
            "exchanges_accepted": self.exchanges_accepted,
            "exchange_history": self.exchange_history,
        }

        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

    def save_results(self):
        """Save final simulation results."""
        results_file = self.output_dir / "remd_results.pkl"
        results = {
            "temperatures": self.temperatures,
            "n_replicas": self.n_replicas,
            "exchange_frequency": self.exchange_frequency,
            "exchange_attempts": self.exchange_attempts,
            "exchanges_accepted": self.exchanges_accepted,
            "final_acceptance_rate": self.exchanges_accepted
            / max(1, self.exchange_attempts),
            "replica_states": self.replica_states,
            "state_replicas": self.state_replicas,
            "exchange_history": self.exchange_history,
            "trajectory_files": [str(f) for f in self.trajectory_files],
        }

        with open(results_file, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Results saved to {results_file}")

    def demux_trajectories(
        self, target_temperature: float = 300.0, equilibration_steps: int = 100
    ) -> Optional[
        str
    ]:  # Fixed: Changed return type to Optional[str] to allow None returns
        """
        Demultiplex trajectories to extract frames at target temperature.

        Args:
            target_temperature: Target temperature to extract frames for
            equilibration_steps: Number of equilibration steps (needed for frame calculation)

        Returns:
            Path to the demultiplexed trajectory file, or None if failed
        """
        logger.info(f"Demultiplexing trajectories for T = {target_temperature} K")

        # Find the target temperature index
        target_temp_idx = np.argmin(
            np.abs(np.array(self.temperatures) - target_temperature)
        )
        actual_temp = self.temperatures[target_temp_idx]

        logger.info(f"Using closest temperature: {actual_temp:.1f} K")

        # Check if we have exchange history
        if not self.exchange_history:
            logger.warning("No exchange history available for demultiplexing")
            return None

        # DCD reporter settings (must match setup_replicas)
        dcd_frequency = int(max(1, getattr(self, "dcd_stride", 1000)))

        # Load all trajectories with proper error handling
        demux_frames = []
        trajectory_frame_counts = {}  # Cache frame counts to avoid repeated loading

        logger.info(f"Processing {len(self.exchange_history)} exchange steps...")
        logger.info(
            (
                f"Exchange frequency: {self.exchange_frequency} MD steps, "
                f"DCD frequency: {dcd_frequency} MD steps"
            )
        )

        # Debug: Check files exist and get basic info
        logger.info("DCD File Diagnostics:")
        for i, traj_file in enumerate(self.trajectory_files):
            if traj_file.exists():
                file_size = traj_file.stat().st_size
                logger.info(
                    (
                        f"  Replica {i}: {traj_file.name} exists, size: "
                        f"{file_size:,} bytes"
                    )
                )

                # Try a simple frame count using mdtraj
                try:
                    temp_traj = md.load(str(traj_file), top=self.pdb_file)
                    actual_frames = temp_traj.n_frames
                    logger.info(f"    -> Successfully loaded: {actual_frames} frames")
                    trajectory_frame_counts[str(traj_file)] = actual_frames
                except Exception as e:
                    logger.warning(f"    -> Failed to load: {e}")
                    trajectory_frame_counts[str(traj_file)] = 0
            else:
                logger.warning(f"  Replica {i}: {traj_file.name} does not exist")

        # Plan frames per replica for batched loading
        steps_plan: List[Tuple[int, int]] = []
        frames_by_replica: Dict[int, set] = {}
        for step, replica_states in enumerate(self.exchange_history):
            try:
                replica_at_target = None
                for replica_idx, temp_state in enumerate(replica_states):
                    if temp_state == target_temp_idx:
                        replica_at_target = replica_idx
                        break
                if replica_at_target is None:
                    continue
                md_step = equilibration_steps + step * self.exchange_frequency
                frame_number = md_step // dcd_frequency
                steps_plan.append((replica_at_target, frame_number))
                frames_by_replica.setdefault(replica_at_target, set()).add(frame_number)
            except Exception:
                continue

        # Load each replica once and extract requested frames
        per_replica_frames: Dict[int, Dict[int, md.Trajectory]] = {}
        for replica_idx, frame_set in frames_by_replica.items():
            traj_file = self.trajectory_files[replica_idx]
            if not traj_file.exists():
                logger.warning(f"Trajectory file not found: {traj_file}")
                continue
            try:
                traj = md.load(str(traj_file), top=self.pdb_file)
                n_frames = traj.n_frames
                trajectory_frame_counts[str(traj_file)] = n_frames
                selected: Dict[int, md.Trajectory] = {}
                for fidx in sorted(frame_set):
                    if 0 <= fidx < n_frames:
                        selected[fidx] = traj[fidx]
                per_replica_frames[replica_idx] = selected
            except Exception as e:
                logger.warning(
                    f"Could not load trajectory for replica {replica_idx}: {e}"
                )
                continue

        # Assemble in chronological order
        for replica_idx, frame_number in steps_plan:
            frame_map = per_replica_frames.get(replica_idx, {})
            if frame_number in frame_map:
                demux_frames.append(frame_map[frame_number])

        if demux_frames:
            try:
                # Combine all frames
                demux_traj = md.join(demux_frames)

                # Save demultiplexed trajectory
                demux_file = self.output_dir / f"demux_T{actual_temp:.0f}K.dcd"
                demux_traj.save_dcd(str(demux_file))

                logger.info(f"Demultiplexed trajectory saved: {demux_file}")
                logger.info(f"Total frames at target temperature: {len(demux_frames)}")

                return str(demux_file)
            except Exception as e:
                logger.error(f"Error saving demultiplexed trajectory: {e}")
                return None
        else:
            logger.debug(
                (
                    "No frames found for demultiplexing - this may indicate "
                    "frame indexing issues"
                )
            )
            logger.debug("Debug info:")
            logger.debug(f"  Exchange steps: {len(self.exchange_history)}")
            logger.debug(f"  Exchange frequency: {self.exchange_frequency}")
            logger.debug(f"  Equilibration steps: {equilibration_steps}")
            logger.debug(f"  DCD frequency: {dcd_frequency}")
            for i, traj_file in enumerate(self.trajectory_files):
                n_frames = trajectory_frame_counts.get(str(traj_file), 0)
                logger.debug(f"  Replica {i}: {n_frames} frames in {traj_file.name}")
            return None

    def get_exchange_statistics(self) -> Dict[str, Any]:
        """Get exchange statistics and diagnostics."""
        if not self.exchange_history:
            return {}

        # Calculate mixing statistics
        replica_visits = np.zeros((self.n_replicas, self.n_replicas))
        for states in self.exchange_history:
            for replica, state in enumerate(states):
                replica_visits[replica, state] += 1

        # Normalize to get probabilities
        replica_probs = replica_visits / len(self.exchange_history)

        # Calculate round-trip times (simplified)
        round_trip_times = []
        for replica in range(self.n_replicas):
            # Find when replica returns to its starting state
            start_state = 0  # Assuming replica starts at its own temperature
            current_state = start_state
            trip_start = 0

            for step, states in enumerate(self.exchange_history):
                if states[replica] != current_state:
                    current_state = states[replica]
                    if current_state == start_state and step > trip_start:
                        round_trip_times.append(step - trip_start)
                        trip_start = step

        # Per-pair acceptance rates
        per_pair_acceptance = {}
        for k, att in self.pair_attempt_counts.items():
            acc = self.pair_accept_counts.get(k, 0)
            rate = acc / max(1, att)
            per_pair_acceptance[f"{k}"] = rate

        return {
            "total_exchange_attempts": self.exchange_attempts,
            "total_exchanges_accepted": self.exchanges_accepted,
            "overall_acceptance_rate": self.exchanges_accepted
            / max(1, self.exchange_attempts),
            "replica_state_probabilities": replica_probs.tolist(),
            "average_round_trip_time": (
                np.mean(round_trip_times) if round_trip_times else 0
            ),
            "round_trip_times": round_trip_times[:10],  # First 10 for brevity
            "per_pair_acceptance": per_pair_acceptance,
        }


def setup_bias_variables(pdb_file: str) -> List:
    """
    Set up bias variables for metadynamics.

    Args:
        pdb_file: Path to the PDB file

    Returns:
        List of bias variables
    """
    import mdtraj as md
    from openmm import CustomTorsionForce
    from openmm.app.metadynamics import BiasVariable

    # Load trajectory to get dihedral indices
    traj0 = md.load_pdb(pdb_file)
    phi_indices, _ = md.compute_phi(traj0)

    if len(phi_indices) == 0:
        logger.warning("No phi dihedrals found - proceeding without bias variables")
        return []

    bias_variables = []

    # Add phi dihedral as bias variable
    for i, phi_atoms in enumerate(phi_indices[:2]):  # Use first 2 phi dihedrals
        phi_atoms = [int(atom) for atom in phi_atoms]

        phi_force = CustomTorsionForce("theta")
        phi_force.addTorsion(*phi_atoms, [])

        phi_cv = BiasVariable(
            phi_force,
            -np.pi,  # minValue
            np.pi,  # maxValue
            0.35,  # biasWidth (~20 degrees)
            True,  # periodic
        )

        bias_variables.append(phi_cv)
        logger.info(f"Added phi dihedral {i+1} as bias variable: atoms {phi_atoms}")

    return bias_variables


# Example usage function
def run_remd_simulation(
    pdb_file: str,
    output_dir: str = "output/replica_exchange",
    total_steps: int = 1000,  # VERY FAST for testing
    equilibration_steps: int = 100,  # Default equilibration steps
    temperatures: Optional[List[float]] = None,
    use_metadynamics: bool = True,
    checkpoint_manager=None,
) -> Optional[str]:  # Fixed: Changed return type to Optional[str] to allow None returns
    """
    Convenience function to run a complete REMD simulation.

    Args:
        pdb_file: Path to prepared PDB file
        output_dir: Directory for output files
        total_steps: Total simulation steps
        equilibration_steps: Number of equilibration steps before production
        temperatures: Temperature ladder (auto-generated if None)
        use_metadynamics: Whether to use metadynamics biasing
        checkpoint_manager: CheckpointManager instance for state tracking

    Returns:
        Path to demultiplexed trajectory at 300K, or None if failed
    """

    # Stage: Replica Initialization
    if checkpoint_manager and not checkpoint_manager.is_step_completed(
        "replica_initialization"
    ):
        checkpoint_manager.mark_step_started("replica_initialization")

    # Set up bias variables if requested
    bias_variables = setup_bias_variables(pdb_file) if use_metadynamics else None

    # Create and configure REMD
    remd = ReplicaExchange(
        pdb_file=pdb_file,
        temperatures=temperatures,
        output_dir=output_dir,
        exchange_frequency=50,  # Very frequent exchanges for testing
    )

    # Set up replicas
    remd.setup_replicas(bias_variables=bias_variables)

    # Save state
    if checkpoint_manager:
        checkpoint_manager.save_state(
            {
                "remd_config": {
                    "pdb_file": pdb_file,
                    "temperatures": remd.temperatures,
                    "output_dir": output_dir,
                    "exchange_frequency": remd.exchange_frequency,
                    "bias_variables": bias_variables,
                }
            }
        )
        checkpoint_manager.mark_step_completed(
            "replica_initialization",
            {
                "n_replicas": remd.n_replicas,
                "temperature_range": f"{min(remd.temperatures):.1f}-{max(remd.temperatures):.1f}K",
            },
        )
    elif checkpoint_manager and checkpoint_manager.is_step_completed(
        "replica_initialization"
    ):
        # Load existing state
        state_data = checkpoint_manager.load_state()
        remd_config = state_data.get("remd_config", {})

        # Recreate REMD object
        remd = ReplicaExchange(
            pdb_file=pdb_file,
            temperatures=temperatures,
            output_dir=output_dir,
            exchange_frequency=50,
        )

        # Set up bias variables if they were used
        bias_variables = remd_config.get("bias_variables") if use_metadynamics else None

        # Only setup replicas if we haven't done energy minimization yet
        if not checkpoint_manager.is_step_completed("energy_minimization"):
            remd.setup_replicas(bias_variables=bias_variables)
    else:
        # Non-checkpoint mode (legacy)
        bias_variables = setup_bias_variables(pdb_file) if use_metadynamics else None
        remd = ReplicaExchange(
            pdb_file=pdb_file,
            temperatures=temperatures,
            output_dir=output_dir,
            exchange_frequency=50,
        )
        remd.setup_replicas(bias_variables=bias_variables)

    # Run simulation with checkpoint integration
    remd.run_simulation(
        total_steps=total_steps,
        equilibration_steps=equilibration_steps,
        checkpoint_manager=checkpoint_manager,
    )

    # Demultiplex for analysis (separate step - don't fail the entire simulation)
    demux_traj = None
    if checkpoint_manager and not checkpoint_manager.is_step_completed(
        "trajectory_demux"
    ):
        if checkpoint_manager:
            checkpoint_manager.mark_step_started("trajectory_demux")

        # Small delay to ensure DCD files are fully written to disk
        import time

        logger.info("Waiting for DCD files to be fully written...")
        time.sleep(2.0)

        try:
            demux_traj = remd.demux_trajectories(
                target_temperature=300.0, equilibration_steps=equilibration_steps
            )
            if demux_traj:
                logger.info(f"✓ Demultiplexing successful: {demux_traj}")
                if checkpoint_manager:
                    checkpoint_manager.mark_step_completed(
                        "trajectory_demux", {"demux_file": demux_traj}
                    )
            else:
                logger.warning("⚠ Demultiplexing returned no trajectory")
                if checkpoint_manager:
                    checkpoint_manager.mark_step_failed(
                        "trajectory_demux", "No frames found for demultiplexing"
                    )
        except Exception as e:
            logger.warning(f"⚠ Demultiplexing failed: {e}")
            if checkpoint_manager:
                checkpoint_manager.mark_step_failed("trajectory_demux", str(e))

        # Always log that the simulation itself was successful
        logger.info("🎉 REMD simulation completed successfully!")
        logger.info("Raw trajectory files are available for manual analysis")
    else:
        logger.info(
            "Trajectory demux already completed or checkpoint manager not available"
        )

    # Print statistics
    stats = remd.get_exchange_statistics()
    logger.info(f"REMD Statistics: {stats}")

    return demux_traj
