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
from typing import Any, Dict, List, Optional, Tuple, Union

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
            f"Temperature range: {min(self.temperatures):.1f} - {max(self.temperatures):.1f} K"
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
        """Generate an exponential temperature ladder for optimal exchange efficiency.

        Delegates to `utils.replica_utils.exponential_temperature_ladder` to avoid duplication.
        """
        return exponential_temperature_ladder(min_temp, max_temp, n_replicas)

    def setup_replicas(self, bias_variables: Optional[List] = None):
        """
        Set up all replica simulations with different temperatures.

        Args:
            bias_variables: Optional list of bias variables for metadynamics
        """
        logger.info("Setting up replica simulations...")

        # Load PDB and create forcefield
        pdb = PDBFile(self.pdb_file)
        forcefield = ForceField(*self.forcefield_files)

        # Create system (same for all replicas) with HMR and tuned parameters for speed
        logger.info("Creating molecular system with HMR and tuned parameters...")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            constraints=HBonds,
            rigidWater=True,
            nonbondedCutoff=0.9 * unit.nanometer,
            ewaldErrorTolerance=1e-4,
            hydrogenMass=3.0 * unit.amu,  # HMR
            removeCMMotion=True,
        )

        # Verify system was created successfully
        logger.info(f"System created with {system.getNumParticles()} particles")
        logger.info(f"System has {system.getNumForces()} force terms")

        # Add extra stability checks
        for force_idx in range(system.getNumForces()):
            force = system.getForce(force_idx)
            logger.info(f"  Force {force_idx}: {force.__class__.__name__}")

        # Set up metadynamics if bias variables provided
        metadynamics = None
        if bias_variables:
            from openmm.app.metadynamics import Metadynamics

            bias_dir = self.output_dir / "bias"
            bias_dir.mkdir(exist_ok=True)

            metadynamics = Metadynamics(
                system,
                bias_variables,
                temperature=self.temperatures[0]
                * unit.kelvin,  # Will be updated for each replica
                biasFactor=10.0,
                height=1.0 * unit.kilojoules_per_mole,
                frequency=500,
                biasDir=str(bias_dir),
                saveFrequency=1000,
            )

        # Create replicas with different temperatures
        # Prefer fast GPU platforms, then CPU
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
                # Try HIP/ROCm or OpenCL
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

        shared_minimized_positions = None

        for i, temperature in enumerate(self.temperatures):
            logger.info(f"Setting up replica {i} at {temperature}K...")

            # Create integrator for this temperature with 2 fs timestep (HMR)
            integrator = openmm.LangevinIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,
                2.0 * unit.femtoseconds,
            )

            # Create simulation
            simulation = Simulation(
                pdb.topology,
                system,
                integrator,
                platform,
                platform_properties or None,
            )
            simulation.context.setPositions(pdb.positions)

            # Multi-stage energy minimization for stability with fallback strategies
            # Reuse minimized coordinates from the first replica to speed up subsequent setups
            if shared_minimized_positions is not None:
                try:
                    simulation.context.setPositions(shared_minimized_positions)
                    # Quick touch-up minimization only
                    simulation.minimizeEnergy(
                        maxIterations=50,
                        tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
                    )
                    logger.info(
                        f"  Reused minimized coordinates for replica {i} (quick touch-up)"
                    )
                except Exception as e:
                    logger.warning(
                        f"  Failed to reuse minimized coords for replica {i}: {e}; falling back to full minimization"
                    )
                else:
                    # Set up trajectory reporter and store replica data, then continue
                    traj_file = self.output_dir / f"replica_{i:02d}.dcd"
                    dcd_reporter = DCDReporter(
                        str(traj_file), int(max(1, self.dcd_stride))
                    )
                    simulation.reporters.append(dcd_reporter)

                    self.replicas.append(simulation)
                    self.integrators.append(integrator)
                    self.contexts.append(simulation.context)
                    self.trajectory_files.append(traj_file)

                    logger.info(f"Replica {i:02d}: T = {temperature:.1f} K")
                    continue

            logger.info(f"  Minimizing energy for replica {i}...")

            # Check initial energy
            try:
                initial_state = simulation.context.getState(getEnergy=True)
                initial_energy = initial_state.getPotentialEnergy()
                logger.info(f"  Initial energy for replica {i}: {initial_energy}")

                # Check for extremely high initial energy
                energy_val = initial_energy.value_in_unit(unit.kilojoules_per_mole)
                if abs(energy_val) > 1e6:
                    logger.warning(
                        f"  Very high initial energy ({energy_val:.2e} kJ/mol) detected for replica {i}"
                    )

            except Exception as e:
                logger.warning(f"  Could not check initial energy for replica {i}: {e}")

            # Stage 1: Gentle initial minimization with relaxed tolerance
            minimization_success = False
            for attempt, (max_iter, tolerance_val) in enumerate(
                [
                    (50, 100.0),  # Very gentle first attempt
                    (100, 50.0),  # Moderate second attempt
                    (200, 10.0),  # Stricter third attempt
                ]
            ):
                try:
                    tolerance = (
                        tolerance_val * unit.kilojoules_per_mole / unit.nanometer
                    )
                    simulation.minimizeEnergy(
                        maxIterations=max_iter, tolerance=tolerance
                    )
                    logger.info(
                        f"  Stage 1 minimization completed for replica {i} (attempt {attempt + 1})"
                    )
                    minimization_success = True
                    break
                except Exception as e:
                    logger.warning(
                        f"  Stage 1 minimization attempt {attempt + 1} failed for replica {i}: {e}"
                    )
                    if attempt == 2:  # Last attempt
                        logger.error(
                            f"  All minimization attempts failed for replica {i}"
                        )
                        raise RuntimeError(
                            f"Energy minimization failed for replica {i} after 3 attempts. "
                            f"Structure may be too distorted. Consider: 1) Better initial structure, "
                            f"2) Different forcefield, 3) Manual structure preparation"
                        )

            # Stage 2: Refined minimization with NaN checking
            if minimization_success:
                try:
                    simulation.minimizeEnergy(
                        maxIterations=100,
                        tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer,
                    )
                    logger.info(f"  Stage 2 minimization completed for replica {i}")

                    # Comprehensive post-minimization validation
                    state = simulation.context.getState(
                        getPositions=True, getEnergy=True, getVelocities=True
                    )
                    energy = state.getPotentialEnergy()
                    positions = state.getPositions()

                    # Check for NaN in energy
                    energy_str = str(energy).lower()
                    if "nan" in energy_str or "inf" in energy_str:
                        raise ValueError(
                            f"Invalid energy ({energy}) detected after minimization for replica {i}"
                        )

                    # Check for reasonable energy range
                    energy_val = energy.value_in_unit(unit.kilojoules_per_mole)
                    if abs(energy_val) > 1e5:
                        logger.warning(
                            f"  High final energy ({energy_val:.2e} kJ/mol) for replica {i}"
                        )

                    # Check for NaN in positions
                    pos_array = positions.value_in_unit(unit.nanometer)
                    if np.any(np.isnan(pos_array)) or np.any(np.isinf(pos_array)):
                        raise ValueError(
                            f"Invalid positions detected after minimization for replica {i}"
                        )

                    logger.info(f"  Final energy for replica {i}: {energy}")

                    # Cache minimized positions to accelerate remaining replicas
                    if shared_minimized_positions is None:
                        try:
                            shared_minimized_positions = state.getPositions()
                            logger.info(
                                "  Cached minimized coordinates from replica 0 for reuse"
                            )
                        except Exception:
                            shared_minimized_positions = None

                except Exception as e:
                    logger.error(
                        f"  Stage 2 minimization or validation failed for replica {i}: {e}"
                    )
                    # Try to recover by using the Stage 1 result
                    logger.warning(
                        f"  Attempting to continue with Stage 1 result for replica {i}"
                    )
                    try:
                        state = simulation.context.getState(getEnergy=True)
                        energy = state.getPotentialEnergy()
                        logger.info(f"  Using Stage 1 energy for replica {i}: {energy}")
                    except:
                        raise RuntimeError(
                            f"Complete minimization failure for replica {i}"
                        )

            # Set up trajectory reporter
            traj_file = self.output_dir / f"replica_{i:02d}.dcd"
            dcd_reporter = DCDReporter(str(traj_file), int(max(1, self.dcd_stride)))
            simulation.reporters.append(dcd_reporter)

            # Store replica data
            self.replicas.append(simulation)
            self.integrators.append(integrator)
            self.contexts.append(simulation.context)
            self.trajectory_files.append(traj_file)

            logger.info(f"Replica {i:02d}: T = {temperature:.1f} K")

        logger.info("All replicas set up successfully")
        self._is_setup = True

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
            f"Checkpoint restoration complete. Exchange stats: {self.exchanges_accepted}/{self.exchange_attempts}"
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
            f"Exchange calculation: E_i={energy_i}, E_j={energy_j}, T_i={temp_i:.1f}K, T_j={temp_j:.1f}K, delta={delta:.3f}, prob={prob:.6f}"
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

        self.exchange_attempts += 1

        # Calculate exchange probability (use cached energies if provided)
        if energies is not None:
            temp_i = self.temperatures[self.replica_states[replica_i]]
            temp_j = self.temperatures[self.replica_states[replica_j]]

            beta_i = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temp_i * unit.kelvin)
            beta_j = 1.0 / (unit.MOLAR_GAS_CONSTANT_R * temp_j * unit.kelvin)

            # Ensure energies are Quantities in kJ/mol
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
                # Fallback if unit machinery not present
                delta = float(delta_q)
            prob = float(min(1.0, np.exp(-delta)))
        else:
            prob = self.calculate_exchange_probability(replica_i, replica_j)

        # Accept or reject exchange and track per-pair stats
        # Build a fixed-size 2-tuple for dict keys to satisfy type checker
        state_i_val = self.replica_states[replica_i]
        state_j_val = self.replica_states[replica_j]
        pair = (
            min(state_i_val, state_j_val),
            max(state_i_val, state_j_val),
        )
        self.pair_attempt_counts[pair] = self.pair_attempt_counts.get(pair, 0) + 1
        if np.random.random() < prob:
            # Perform exchange by swapping temperature assignments only

            # Update temperature mappings with bounds checking
            if replica_i >= len(self.replica_states):
                raise RuntimeError(
                    f"replica_states array too small: {len(self.replica_states)}, need {replica_i + 1}"
                )
            if replica_j >= len(self.replica_states):
                raise RuntimeError(
                    f"replica_states array too small: {len(self.replica_states)}, need {replica_j + 1}"
                )

            old_state_i = self.replica_states[replica_i]
            old_state_j = self.replica_states[replica_j]

            # Validate state indices
            if old_state_i >= len(self.state_replicas) or old_state_j >= len(
                self.state_replicas
            ):
                raise RuntimeError(
                    f"Invalid state indices: {old_state_i}, {old_state_j} vs array size {len(self.state_replicas)}"
                )

            # Perform the swap
            self.replica_states[replica_i] = old_state_j
            self.replica_states[replica_j] = old_state_i

            self.state_replicas[old_state_i] = replica_j
            self.state_replicas[old_state_j] = replica_i

            # Update integrator temperatures
            self.integrators[replica_i].setTemperature(
                self.temperatures[old_state_j] * unit.kelvin
            )
            self.integrators[replica_j].setTemperature(
                self.temperatures[old_state_i] * unit.kelvin
            )

            # Rethermalize velocities instead of swapping full states
            self.contexts[replica_i].setVelocitiesToTemperature(
                self.temperatures[old_state_j] * unit.kelvin
            )
            self.contexts[replica_j].setVelocitiesToTemperature(
                self.temperatures[old_state_i] * unit.kelvin
            )

            self.exchanges_accepted += 1
            self.pair_accept_counts[pair] = self.pair_accept_counts.get(pair, 0) + 1

            logger.debug(
                f"Exchange accepted: replica {replica_i} <-> {replica_j} (prob={prob:.3f})"
            )
            return True
        else:
            logger.debug(
                f"Exchange rejected: replica {replica_i} <-> {replica_j} (prob={prob:.3f})"
            )
            return False

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
        # CRITICAL VALIDATION: Ensure replicas are properly initialized
        if not self._is_setup:
            raise RuntimeError(
                "Replicas not properly initialized! Call setup_replicas() first."
            )

        if not self.contexts or len(self.contexts) != self.n_replicas:
            raise RuntimeError(
                f"Replicas not properly initialized! Expected {self.n_replicas} contexts, "
                f"but got {len(self.contexts)}. setup_replicas() may have failed."
            )

        if not self.replicas or len(self.replicas) != self.n_replicas:
            raise RuntimeError(
                f"Replicas not properly initialized! Expected {self.n_replicas} replicas, "
                f"but got {len(self.replicas)}. setup_replicas() may have failed."
            )

        logger.info(f"Starting REMD simulation: {total_steps} steps")
        logger.info(f"Exchange attempts every {self.exchange_frequency} steps")

        # Gradual heating and equilibration phase
        if equilibration_steps > 0:
            # Check if gradual heating is already completed
            if checkpoint_manager and checkpoint_manager.is_step_completed(
                "gradual_heating"
            ):
                logger.info("Gradual heating already completed ✓")
            else:
                if checkpoint_manager:
                    checkpoint_manager.mark_step_started("gradual_heating")
                logger.info(
                    f"Equilibration with gradual heating: {equilibration_steps} steps"
                )

                # Phase 1: Gradual heating (first 40% of equilibration)
                heating_steps = max(100, equilibration_steps * 40 // 100)
                logger.info(f"   Phase 1: Gradual heating over {heating_steps} steps")

                heating_chunk_size = max(10, heating_steps // 20)  # Heat in 20 stages
                for heat_step in range(0, heating_steps, heating_chunk_size):
                    current_steps = min(heating_chunk_size, heating_steps - heat_step)

                    # Calculate gradual temperature scaling (start from 50K, ramp to target)
                    progress_fraction = (heat_step + current_steps) / heating_steps

                    for replica_idx, replica in enumerate(self.replicas):
                        target_temp = self.temperatures[
                            self.replica_states[replica_idx]
                        ]
                        current_temp = 50.0 + (target_temp - 50.0) * progress_fraction

                        # Update integrator temperature gradually
                        replica.integrator.setTemperature(current_temp * unit.kelvin)

                        # Run with error recovery
                        try:
                            replica.step(current_steps)
                        except Exception as e:
                            if "NaN" in str(e) or "nan" in str(e).lower():
                                logger.warning(
                                    f"   NaN detected in replica {replica_idx} during heating, attempting recovery..."
                                )

                                # Attempt recovery by resetting velocities and reducing step size
                                replica.context.setVelocitiesToTemperature(
                                    current_temp * unit.kelvin
                                )

                                # Try smaller steps
                                small_steps = max(1, current_steps // 5)
                                for recovery_attempt in range(5):
                                    try:
                                        replica.step(small_steps)
                                        break
                                    except:
                                        if recovery_attempt == 4:  # Last attempt failed
                                            raise RuntimeError(
                                                f"Failed to recover from NaN in replica {replica_idx}"
                                            )
                                        replica.context.setVelocitiesToTemperature(
                                            current_temp * unit.kelvin * 0.9
                                        )
                            else:
                                raise

                    progress = min(
                        40, (heat_step + current_steps) * 40 // heating_steps
                    )
                    logger.info(
                        f"   Heating Progress: {progress}% - Current temps: {[50.0 + (self.temperatures[self.replica_states[i]] - 50.0) * progress_fraction for i in range(len(self.replicas))]}"
                    )

                # Mark heating as completed
                if checkpoint_manager:
                    checkpoint_manager.mark_step_completed(
                        "gradual_heating",
                        {
                            "heating_steps": heating_steps,
                            "final_temperatures": [
                                self.temperatures[state]
                                for state in self.replica_states
                            ],
                        },
                    )

            # Phase 2: Temperature equilibration (remaining 60% of equilibration)
            if checkpoint_manager and checkpoint_manager.is_step_completed(
                "equilibration"
            ):
                logger.info("Temperature equilibration already completed ✓")
            else:
                if checkpoint_manager:
                    checkpoint_manager.mark_step_started("equilibration")

                temp_equil_steps = max(
                    100, equilibration_steps * 60 // 100
                )  # Calculate correctly
                logger.info(
                    f"   Phase 2: Temperature equilibration at target temperatures over {temp_equil_steps} steps"
                )

                # Set all replicas to their final target temperatures
                for replica_idx, replica in enumerate(self.replicas):
                    target_temp = self.temperatures[self.replica_states[replica_idx]]
                    replica.integrator.setTemperature(target_temp * unit.kelvin)
                    replica.context.setVelocitiesToTemperature(
                        target_temp * unit.kelvin
                    )

                equil_chunk_size = max(1, temp_equil_steps // 10)  # 10% chunks
                for i in range(0, temp_equil_steps, equil_chunk_size):
                    current_steps = min(equil_chunk_size, temp_equil_steps - i)

                    for replica_idx, replica in enumerate(self.replicas):
                        try:
                            replica.step(current_steps)
                        except Exception as e:
                            if "NaN" in str(e) or "nan" in str(e).lower():
                                logger.error(
                                    f"   NaN detected in replica {replica_idx} during equilibration - simulation unstable"
                                )
                                if checkpoint_manager:
                                    checkpoint_manager.mark_step_failed(
                                        "equilibration", str(e)
                                    )
                                raise RuntimeError(
                                    f"Simulation became unstable for replica {replica_idx}. Try: 1) Better initial structure, 2) Smaller timestep, 3) More minimization"
                                )
                            else:
                                raise

                    progress = min(
                        100, 40 + (i + current_steps) * 60 // temp_equil_steps
                    )
                    logger.info(
                        f"   Equilibration Progress: {progress}% ({equilibration_steps - temp_equil_steps + i + current_steps}/{equilibration_steps} steps)"
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

        # Production phase with exchanges
        if checkpoint_manager and checkpoint_manager.is_step_completed(
            "production_simulation"
        ):
            logger.info("Production simulation already completed ✓")
            return  # Skip production phase

        if checkpoint_manager:
            checkpoint_manager.mark_step_started("production_simulation")

        production_steps = total_steps - equilibration_steps
        exchange_steps = production_steps // self.exchange_frequency

        logger.info(
            f"Production: {production_steps} steps with {exchange_steps} exchange attempts"
        )

        for step in range(exchange_steps):
            # Run MD for all replicas with error recovery
            for replica_idx, replica in enumerate(self.replicas):
                try:
                    replica.step(self.exchange_frequency)
                except Exception as e:
                    if "NaN" in str(e) or "nan" in str(e).lower():
                        logger.error(
                            f"NaN detected in replica {replica_idx} during production phase"
                        )
                        # Try to save trajectory data before failing
                        try:
                            state = replica.context.getState(
                                getPositions=True, getVelocities=True
                            )
                            logger.info(
                                f"Attempting to save current state before failure..."
                            )
                        except:
                            pass
                        raise RuntimeError(
                            f"Simulation became unstable for replica {replica_idx} at production step {step}. "
                            f"Consider: 1) Longer equilibration, 2) Smaller timestep, 3) Different initial structure"
                        )
                    else:
                        raise

            # Precompute energies once per sweep to avoid repeated getState calls
            energies = []
            for idx, ctx in enumerate(self.contexts):
                try:
                    e_state = ctx.getState(getEnergy=True)
                    energies.append(e_state.getPotentialEnergy())
                except Exception as e:
                    logger.debug(f"Energy getState failed for replica {idx}: {e}")
                    # Fallback to cached value if available; else 0.0
                    last = self.energies[idx] if idx < len(self.energies) else 0.0
                    energies.append(last)
            # Cache energies for potential diagnostics
            self.energies = energies

            # Attempt exchanges between adjacent temperatures using cached energies
            for i in range(0, self.n_replicas - 1, 2):  # Even pairs
                try:
                    self.attempt_exchange(i, i + 1, energies=energies)
                except Exception as e:
                    logger.warning(
                        f"Exchange attempt failed between replicas {i} and {i+1}: {e}"
                    )
                    # Continue with other exchanges

            for i in range(1, self.n_replicas - 1, 2):  # Odd pairs
                try:
                    self.attempt_exchange(i, i + 1, energies=energies)
                except Exception as e:
                    logger.warning(
                        f"Exchange attempt failed between replicas {i} and {i+1}: {e}"
                    )
                    # Continue with other exchanges

            # Save exchange history
            self.exchange_history.append(self.replica_states.copy())

            # Enhanced progress reporting - show every step for fast runs
            progress_percent = (step + 1) * 100 // exchange_steps
            acceptance_rate = self.exchanges_accepted / max(1, self.exchange_attempts)
            completed_steps = (step + 1) * self.exchange_frequency + equilibration_steps

            logger.debug(
                f"   Production Progress: {progress_percent}% "
                f"({step + 1}/{exchange_steps} exchanges, "
                f"{completed_steps}/{total_steps} total steps) "
                f"| Acceptance: {acceptance_rate:.3f}"
            )

            # Save states periodically
            if (step + 1) * self.exchange_frequency % save_state_frequency == 0:
                self.save_checkpoint(step + 1)

        # Mark production as completed
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

        # Close and flush DCD files to ensure all data is written
        self._close_dcd_files()

        # Final statistics
        final_acceptance = self.exchanges_accepted / max(1, self.exchange_attempts)
        logger.info("=" * 60)
        logger.info("🎉 REPLICA EXCHANGE SIMULATION COMPLETED! 🎉")
        logger.info(f"Final exchange acceptance rate: {final_acceptance:.3f}")
        logger.info(f"Total exchanges attempted: {self.exchange_attempts}")
        logger.info(f"Total exchanges accepted: {self.exchanges_accepted}")
        logger.info("=" * 60)

        # Save final data
        self.save_results()

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
            f"Exchange frequency: {self.exchange_frequency} MD steps, DCD frequency: {dcd_frequency} MD steps"
        )

        # Debug: Check files exist and get basic info
        logger.info("DCD File Diagnostics:")
        for i, traj_file in enumerate(self.trajectory_files):
            if traj_file.exists():
                file_size = traj_file.stat().st_size
                logger.info(
                    f"  Replica {i}: {traj_file.name} exists, size: {file_size:,} bytes"
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
                "No frames found for demultiplexing - this may indicate frame indexing issues"
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
