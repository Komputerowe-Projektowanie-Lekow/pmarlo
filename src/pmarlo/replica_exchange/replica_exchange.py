# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Replica Exchange Molecular Dynamics (REMD) implementation for enhanced sampling.

This module provides functionality to run replica exchange simulations using OpenMM,
allowing for better exploration of conformational space across multiple temperatures.
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import openmm
from openmm import Platform, unit
from openmm.app import ForceField, PDBFile, Simulation

from pmarlo.progress import ProgressCB, ProgressReporter
from pmarlo.utils.progress import ProgressPrinter

from ..results import REMDResult
from ..utils.integrator import create_langevin_integrator
from ..utils.naming import base_shape_str, permutation_name
from ..utils.replica_utils import exponential_temperature_ladder
from .config import RemdConfig
from .demux_metadata import DemuxIntegrityError, DemuxMetadata
from .diagnostics import compute_exchange_statistics, retune_temperature_ladder
from .platform_selector import select_platform_and_properties
from .system_builder import (
    create_system,
    load_pdb_and_forcefield,
    log_system_info,
    setup_metadynamics,
)
from .trajectory import ClosableDCDReporter

logger = logging.getLogger("pmarlo")


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
        dcd_stride: int = 1,
        target_accept: float = 0.30,
        config: Optional[RemdConfig] = None,
        random_seed: Optional[int] = None,
        random_state: int | None = None,
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
            target_accept: Desired per-pair exchange acceptance probability
            random_state: Seed for deterministic behaviour. ``random_seed`` is
                accepted for backward compatibility and is overridden by
                ``random_state`` when both are provided.
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
        self.target_accept = target_accept
        self.reporter_stride: Optional[int] = None
        self._replica_reporter_stride: List[int] = []
        self.frames_per_replica_target: Optional[int] = None

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Reproducibility: RNG seeding
        if (
            config
            and hasattr(config, "target_frames_per_replica")
            and getattr(config, "target_frames_per_replica", None) is not None
        ):
            try:
                self.frames_per_replica_target = int(
                    getattr(config, "target_frames_per_replica")
                )
            except Exception:
                self.frames_per_replica_target = None
        if config and getattr(config, "random_seed", None) is not None:
            seed = int(getattr(config, "random_seed"))
        elif random_state is not None:
            seed = int(random_state)
        elif random_seed is not None:
            seed = int(random_seed)
        else:
            seed = int.from_bytes(os.urandom(8), "little") & 0x7FFFFFFF

        self.random_seed = seed
        self.rng = np.random.default_rng(seed)

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
        # Diagnostics accumulation
        self.acceptance_matrix: Optional[np.ndarray] = None
        self.replica_visit_counts: Optional[np.ndarray] = None

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
            # Ensure a reporter stride exists for auto-setup
            if self.reporter_stride is None:
                self.reporter_stride = max(1, self.dcd_stride)
                logger.info(
                    f"Reporter stride not planned; defaulting to dcd_stride={self.reporter_stride} for auto_setup"
                )
            self.setup_replicas()

    @classmethod
    def from_config(cls, config: RemdConfig) -> "ReplicaExchange":
        """Construct instance using immutable RemdConfig as single source of truth."""
        return cls(
            pdb_file=config.pdb_file,
            forcefield_files=config.forcefield_files,
            temperatures=config.temperatures,
            output_dir=str(config.output_dir),
            exchange_frequency=config.exchange_frequency,
            auto_setup=config.auto_setup,
            dcd_stride=config.dcd_stride,
            target_accept=config.target_accept,
            config=config,
            random_seed=getattr(config, "random_seed", None),
        )

    def plan_reporter_stride(
        self,
        total_steps: int,
        equilibration_steps: int,
        target_frames: int = 5000,
    ) -> int:
        """Plan and freeze the reporter stride for this run.

        Decide the DCD stride once, before reporters are added, and store it.
        """
        assert self.reporter_stride is None, "reporter_stride already planned"
        production_steps = max(0, total_steps - equilibration_steps)
        stride = max(1, production_steps // max(1, target_frames))
        self.reporter_stride = stride
        return stride

    def plan_runtime(
        self,
        walltime: float,
        throughput_estimator: Callable[[], float] | float,
        transitions_per_state: int = 50,
        n_states: Optional[int] = None,
        equilibration_fraction: float = 0.1,
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """Plan steps, stride and exchange frequency for a walltime budget.

        Parameters
        ----------
        walltime:
            Total wall-clock time budget in seconds.
        throughput_estimator:
            Either a callable returning estimated MD steps per second or a
            numeric value.
        transitions_per_state:
            Minimum effective transitions required per state.
        n_states:
            Number of states; defaults to the number of replicas.
        equilibration_fraction:
            Fraction of total steps reserved for equilibration.
        dry_run:
            If ``True``, only compute and log the plan without mutating
            instance attributes.
        """

        steps_per_second = (
            float(throughput_estimator())
            if callable(throughput_estimator)
            else float(throughput_estimator)
        )
        total_steps = int(max(1, walltime * steps_per_second))
        min_equil = 200
        equilibration_steps = int(max(min_equil, total_steps * equilibration_fraction))
        production_steps = max(0, total_steps - equilibration_steps)
        states = int(n_states or self.n_replicas)
        target_frames = max(1, transitions_per_state * states)
        stride = max(1, production_steps // target_frames)
        exchange_frequency = max(1, production_steps // target_frames)
        expected_frames = production_steps // stride
        plan = {
            "total_steps": total_steps,
            "equilibration_steps": equilibration_steps,
            "exchange_frequency": exchange_frequency,
            "reporter_stride": stride,
            "expected_frames": expected_frames,
        }
        logger.info(
            (
                "Runtime plan: total_steps=%d equilibration=%d stride=%d "
                "exchange_frequency=%d expected_frames=%d"
            ),
            total_steps,
            equilibration_steps,
            stride,
            exchange_frequency,
            expected_frames,
        )
        if not dry_run:
            self.exchange_frequency = exchange_frequency
            self.plan_reporter_stride(
                total_steps, equilibration_steps, target_frames=target_frames
            )
        return plan

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
        # Enforce stride planning before creating reporters
        assert (
            self.reporter_stride is not None and self.reporter_stride > 0
        ), "reporter_stride is not planned. Call plan_reporter_stride(...) before setup_replicas()"

        pdb, forcefield = load_pdb_and_forcefield(self.pdb_file, self.forcefield_files)
        system = create_system(pdb, forcefield)
        log_system_info(system, logger)
        self.metadynamics = setup_metadynamics(
            system, bias_variables, self.temperatures[0], self.output_dir
        )
        platform, platform_properties = select_platform_and_properties(logger)

        shared_minimized_positions = None

        for i, temperature in enumerate(self.temperatures):
            logger.info(f"Setting up replica {i} at {temperature}K...")

            integrator = self._create_integrator_for_temperature(temperature)
            # Offset integrator seed per replica
            try:
                integrator.setRandomNumberSeed(int(self.random_seed + i))
            except Exception:
                pass
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

    def _load_pdb_and_forcefield(self) -> Tuple[PDBFile, ForceField]:  # Deprecated
        return load_pdb_and_forcefield(self.pdb_file, self.forcefield_files)

    def _create_system(
        self, pdb: PDBFile, forcefield: ForceField
    ) -> openmm.System:  # Deprecated
        return create_system(pdb, forcefield)

    def _log_system_info(self, system: openmm.System) -> None:  # Deprecated
        return log_system_info(system, logger)

    def _setup_metadynamics(
        self, system: openmm.System, bias_variables: Optional[List]
    ) -> None:  # Deprecated
        self.metadynamics = setup_metadynamics(
            system, bias_variables, self.temperatures[0], self.output_dir
        )

    def _select_platform_and_properties(
        self,
    ) -> Tuple[Platform, Dict[str, str]]:  # Deprecated
        return select_platform_and_properties(logger)

    def _create_integrator_for_temperature(
        self, temperature: float
    ) -> openmm.Integrator:
        return create_langevin_integrator(temperature, self.random_seed)

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
        stride = int(
            self.reporter_stride
            if self.reporter_stride is not None
            else max(1, self.dcd_stride)
        )
        dcd_reporter = ClosableDCDReporter(str(traj_file), stride)
        simulation.reporters.append(dcd_reporter)
        self._replica_reporter_stride.append(stride)
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
            "random_seed": self.random_seed,
            "rng_state": self.rng.bit_generator.state,
        }

        # Save states in XML for long-term stability across versions
        from openmm import XmlSerializer  # type: ignore

        replica_xml_states: List[str] = []
        for i, context in enumerate(self.contexts):
            try:
                sim_state = context.getState(
                    getPositions=True, getVelocities=True, getEnergy=True
                )
                xml_str = XmlSerializer.serialize(sim_state)
                replica_xml_states.append(xml_str)
            except Exception as e:
                logger.warning(f"Could not save state XML for replica {i}: {e}")
                replica_xml_states.append("")

        state["replica_state_xml"] = replica_xml_states
        # Persist reporter stride data for demux after resume
        state["reporter_stride"] = int(self.reporter_stride or max(1, self.dcd_stride))
        state["replica_reporter_strides"] = self._replica_reporter_stride.copy()
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
        # Restore RNG for reproducible continuation
        self.random_seed = checkpoint_state.get("random_seed", self.random_seed)
        rng_state = checkpoint_state.get("rng_state")
        self.rng = np.random.default_rng()
        if rng_state is not None:
            try:
                self.rng.bit_generator.state = rng_state
            except Exception:
                self.rng = np.random.default_rng(self.random_seed)
        else:
            self.rng = np.random.default_rng(self.random_seed)

        # If replicas aren't set up, set them up first
        if not self.is_setup():
            logger.info("Setting up replicas for checkpoint restoration...")
            self.setup_replicas(bias_variables=bias_variables)

        # Restore reporter stride info if present
        self.reporter_stride = checkpoint_state.get(
            "reporter_stride", self.reporter_stride
        )
        saved_replica_strides = checkpoint_state.get("replica_reporter_strides")
        if isinstance(saved_replica_strides, list):
            try:
                self._replica_reporter_stride = [int(x) for x in saved_replica_strides]
            except Exception:
                pass

        # Restore replica states from XML if available
        from openmm import XmlSerializer  # type: ignore

        replica_xml = checkpoint_state.get("replica_state_xml", [])
        if replica_xml and len(replica_xml) == self.n_replicas:
            logger.info("Restoring individual replica states from XML...")
            for i, (context, xml_str) in enumerate(zip(self.contexts, replica_xml)):
                if xml_str:
                    try:
                        state_obj = XmlSerializer.deserialize(xml_str)
                        if state_obj.getPositions() is not None:
                            context.setPositions(state_obj.getPositions())
                        if state_obj.getVelocities() is not None:
                            context.setVelocities(state_obj.getVelocities())
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
        # Correct Metropolis acceptance for temperature swap
        delta = safe_dimensionless((beta_i - beta_j) * (energy_j - energy_i))
        prob = min(1.0, np.exp(delta))

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

        if self.rng.random() < prob:
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

        delta_q = (beta_i - beta_j) * (e_j - e_i)
        try:
            delta = delta_q.value_in_unit(unit.dimensionless)
        except Exception:
            delta = float(delta_q)
        return float(min(1.0, np.exp(delta)))

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

        # Cache a deterministic name for the new permutation of replicas.
        shape_name = base_shape_str((len(self.replica_states),))
        perm_name = permutation_name(tuple(self.replica_states))
        logger.debug(
            "Replica state permutation %s applied (shape %s)", perm_name, shape_name
        )

        self.state_replicas[old_state_i] = replica_j
        self.state_replicas[old_state_j] = replica_i

        self.integrators[replica_i].setTemperature(
            self.temperatures[old_state_j] * unit.kelvin
        )
        self.integrators[replica_j].setTemperature(
            self.temperatures[old_state_i] * unit.kelvin
        )

        # Rescale velocities deterministically instead of redrawing
        Ti = self.temperatures[old_state_i]
        Tj = self.temperatures[old_state_j]
        scale_ij = float(np.sqrt(max(1e-12, Tj / max(1e-12, Ti))))
        try:
            vi = self.contexts[replica_i].getState(getVelocities=True).getVelocities()
            vj = self.contexts[replica_j].getState(getVelocities=True).getVelocities()
            self.contexts[replica_i].setVelocities(vi * scale_ij)
            self.contexts[replica_j].setVelocities(vj / scale_ij)
        except Exception:
            # Fallback to Maxwell draw if rescaling fails
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
        *,
        progress_callback: ProgressCB | None = None,
        cancel_token: Callable[[], bool] | None = None,
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
        reporter = ProgressReporter(progress_callback)
        reporter.emit("setup", {"message": "initializing"})
        self._log_run_start(total_steps)
        # Decide reporter stride BEFORE production; do not mutate during run
        if self.reporter_stride is None:
            stride = self.plan_reporter_stride(
                total_steps, equilibration_steps, target_frames=5000
            )
            logger.info(f"DCD stride planned as {stride} for ~5000 frames/replica")

        def _should_cancel() -> bool:
            try:
                return bool(cancel_token()) if cancel_token is not None else False
            except Exception:
                return False

        cancelled = False
        if equilibration_steps > 0:
            cancelled = self._run_equilibration_phase(
                equilibration_steps, checkpoint_manager, reporter, _should_cancel
            )
            if cancelled:
                reporter.emit("finished", {"status": "cancelled"})
                return
        if self._skip_production_if_completed(checkpoint_manager):
            return
        self._mark_production_started(checkpoint_manager)
        cancelled = self._run_production_phase(
            total_steps,
            equilibration_steps,
            save_state_frequency,
            checkpoint_manager,
            reporter,
            _should_cancel,
        )
        if cancelled:
            reporter.emit("finished", {"status": "cancelled"})
            return
        self._mark_production_completed(
            total_steps, equilibration_steps, checkpoint_manager
        )
        self._close_dcd_files()
        self._log_final_stats()
        # Announce outputs before saving results (predictable filenames)
        artifacts = [str(p) for p in self.trajectory_files]
        artifacts += [
            str(self.output_dir / "analysis_results.pkl"),
            str(self.output_dir / "analysis_results.json"),
        ]
        reporter.emit("write_output", {"artifacts": artifacts})
        self.save_results()
        reporter.emit("finished", {"status": "ok"})

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
        self,
        equilibration_steps: int,
        checkpoint_manager,
        reporter: ProgressReporter | None,
        should_cancel: Callable[[], bool] | None,
    ) -> bool:
        if checkpoint_manager and checkpoint_manager.is_step_completed(
            "gradual_heating"
        ):
            logger.info("Gradual heating already completed ✓")
        else:
            cancelled = self._run_gradual_heating(
                equilibration_steps, checkpoint_manager, reporter, should_cancel
            )
            if cancelled:
                return True

        if checkpoint_manager and checkpoint_manager.is_step_completed("equilibration"):
            logger.info("Temperature equilibration already completed ✓")
        else:
            cancelled = self._run_temperature_equilibration(
                equilibration_steps, checkpoint_manager, reporter, should_cancel
            )
            if cancelled:
                return True
        return False

    def _run_gradual_heating(
        self,
        equilibration_steps: int,
        checkpoint_manager,
        reporter: ProgressReporter | None,
        should_cancel: Callable[[], bool] | None,
    ) -> bool:
        if checkpoint_manager:
            checkpoint_manager.mark_step_started("gradual_heating")
        logger.info(f"Equilibration with gradual heating: {equilibration_steps} steps")
        heating_steps = max(100, equilibration_steps * 40 // 100)
        logger.info(f"   Phase 1: Gradual heating over {heating_steps} steps")
        heat_progress = ProgressPrinter(heating_steps)
        heating_chunk_size = max(10, heating_steps // 20)
        for heat_step in range(0, heating_steps, heating_chunk_size):
            if should_cancel is not None and should_cancel():
                return True
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
            heat_progress.draw(heat_step + current_steps)
            heat_progress.newline_if_active()
            # Report unified equilibrate progress as fraction of total equilibration
            if reporter is not None:
                cur = min(equilibration_steps, heat_step + current_steps)
                reporter.emit(
                    "equilibrate",
                    {"current_step": cur, "total_steps": int(equilibration_steps)},
                )
            temps_preview = [
                50.0
                + (self.temperatures[self.replica_states[i]] - 50.0) * progress_fraction
                for i in range(len(self.replicas))
            ]
            logger.info(
                f"   Heating Progress: {progress}% - Current temps: {temps_preview}"
            )
        heat_progress.close()
        if should_cancel is not None and should_cancel():
            return True
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
        # Completed without cancellation
        return False

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
        self,
        equilibration_steps: int,
        checkpoint_manager,
        reporter: ProgressReporter | None,
        should_cancel: Callable[[], bool] | None,
    ) -> bool:
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
        temp_progress = ProgressPrinter(temp_equil_steps)
        for i in range(0, temp_equil_steps, equil_chunk_size):
            if should_cancel is not None and should_cancel():
                return True
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
            temp_progress.draw(i + current_steps)
            temp_progress.newline_if_active()
            if reporter is not None:
                heating_steps = max(100, equilibration_steps * 40 // 100)
                cur = min(equilibration_steps, heating_steps + i + current_steps)
                reporter.emit(
                    "equilibrate",
                    {"current_step": cur, "total_steps": int(equilibration_steps)},
                )
            logger.info(
                (
                    f"   Equilibration Progress: {progress}% "
                    f"({equilibration_steps - temp_equil_steps + i + current_steps}/"
                    f"{equilibration_steps} steps)"
                )
            )
        temp_progress.close()
        if should_cancel is not None and should_cancel():
            return True
        if checkpoint_manager:
            checkpoint_manager.mark_step_completed(
                "equilibration",
                {
                    "equilibration_steps": temp_equil_steps,
                    "total_equilibration": equilibration_steps,
                },
            )
        logger.info("   Equilibration Complete ✓")
        return False

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
        reporter: ProgressReporter | None,
        should_cancel: Callable[[], bool] | None,
    ) -> bool:
        production_steps = total_steps - equilibration_steps
        exchange_steps = production_steps // self.exchange_frequency
        logger.info(
            (
                f"Production: {production_steps} steps with "
                f"{exchange_steps} exchange attempts"
            )
        )
        # Initialize diagnostics once production starts
        if self.acceptance_matrix is None:
            self.acceptance_matrix = np.zeros((self.n_replicas - 1, 2), dtype=int)
        if self.replica_visit_counts is None:
            self.replica_visit_counts = np.zeros(
                (self.n_replicas, self.n_replicas), dtype=int
            )

        prod_progress = (
            ProgressPrinter(max(1, exchange_steps)) if exchange_steps > 0 else None
        )
        last_t = time.time()
        for step in range(exchange_steps):
            if should_cancel is not None and should_cancel():
                return True
            self._production_step_all_replicas(step, checkpoint_manager)
            energies = self._precompute_energies()
            self._attempt_all_exchanges(energies)
            self.exchange_history.append(self.replica_states.copy())
            # Update visitation histogram
            for r, s in enumerate(self.replica_states):
                if self.replica_visit_counts is not None:
                    self.replica_visit_counts[r, s] += 1
            self._log_production_progress(
                step, exchange_steps, total_steps, equilibration_steps
            )
            # Unified reporter events
            if reporter is not None:
                # exchange stats after each sweep
                acc_rate = self.exchanges_accepted / max(1, self.exchange_attempts)
                swept = time.time() - last_t
                try:
                    pair_acc = None
                    if self.acceptance_matrix is not None:
                        # rows indexed by pair, col0 attempts, col1 accepts
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rat = self.acceptance_matrix[:, 1] / np.maximum(
                                1, self.acceptance_matrix[:, 0]
                            )
                            pair_acc = [float(x) for x in np.nan_to_num(rat)]
                except Exception:
                    pair_acc = None
                reporter.emit(
                    "exchange",
                    {
                        "sweep_index": int(step + 1),
                        "n_replicas": int(self.n_replicas),
                        "acceptance_mean": float(acc_rate),
                        "step_time_s": round(swept, 3),
                        **(
                            {"acceptance_per_pair": pair_acc}
                            if pair_acc is not None
                            else {}
                        ),
                        "temperatures": [float(t) for t in self.temperatures],
                    },
                )
                # production progress as MD steps
                production_steps = max(0, total_steps - equilibration_steps)
                cur_steps = min((step + 1) * self.exchange_frequency, production_steps)
                reporter.emit(
                    "simulate",
                    {
                        "current_step": int(cur_steps),
                        "total_steps": int(production_steps),
                    },
                )
                last_t = time.time()
            if prod_progress is not None:
                acc_rate = self.exchanges_accepted / max(1, self.exchange_attempts)
                prod_progress.draw(step + 1, suffix=f"acc {acc_rate*100:.1f}%")
                prod_progress.newline_if_active()
            if (step + 1) * self.exchange_frequency % save_state_frequency == 0:
                self.save_checkpoint(step + 1)
        if prod_progress is not None:
            prod_progress.close()
        return False

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
                accepted = self.attempt_exchange(i, i + 1, energies=energies)
                # Update acceptance matrix (even pairs in column 0)
                if self.acceptance_matrix is not None:
                    row = i
                    self.acceptance_matrix[row, 0] += 1  # attempts
                    if accepted:
                        self.acceptance_matrix[row, 1] += 1  # accepts
            except Exception as exc:
                logger.warning(
                    (
                        f"Exchange attempt failed between replicas {i} and {i+1}: "
                        f"{exc}"
                    )
                )
        for i in range(1, self.n_replicas - 1, 2):
            try:
                accepted = self.attempt_exchange(i, i + 1, energies=energies)
                # Update acceptance matrix (odd pairs in next rows)
                if self.acceptance_matrix is not None:
                    row = i
                    self.acceptance_matrix[row, 0] += 1
                    if accepted:
                        self.acceptance_matrix[row, 1] += 1
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
        logger.info("REPLICA EXCHANGE SIMULATION COMPLETED")
        logger.info(f"Final exchange acceptance rate: {final_acceptance:.3f}")
        logger.info(f"Total exchanges attempted: {self.exchange_attempts}")
        logger.info(f"Total exchanges accepted: {self.exchanges_accepted}")
        logger.info("=" * 60)

    def _close_dcd_files(self):
        """Close and flush all DCD files to ensure data is written."""
        logger.info("Closing DCD files...")

        for i, replica in enumerate(self.replicas):
            # Close DCD reporters safely
            dcd_reporters = [
                r for r in replica.reporters if isinstance(r, ClosableDCDReporter)
            ]
            for reporter in dcd_reporters:
                try:
                    reporter.close()
                    logger.debug(f"Closed DCD file for replica {i}")
                except Exception as e:
                    logger.warning(f"Error closing DCD file for replica {i}: {e}")

            # Remove DCD reporters from the simulation
            replica.reporters = [
                r for r in replica.reporters if not isinstance(r, ClosableDCDReporter)
            ]

        # Force garbage collection to ensure file handles are released
        import gc

        gc.collect()

        logger.info("DCD files closed and flushed")

    def save_checkpoint(self, step: int):
        """Save simulation checkpoint."""
        checkpoint_file = self.output_dir / f"checkpoint_step_{step:06d}.pkl"
        checkpoint_data = self.save_checkpoint_state()
        checkpoint_data["step"] = step
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint_data, f)

    def save_results(self) -> None:
        """Save final simulation results."""
        results_file = self.output_dir / "analysis_results.pkl"
        json_file = self.output_dir / "analysis_results.json"
        result = REMDResult(
            temperatures=np.asarray(self.temperatures),
            n_replicas=self.n_replicas,
            exchange_frequency=self.exchange_frequency,
            exchange_attempts=self.exchange_attempts,
            exchanges_accepted=self.exchanges_accepted,
            final_acceptance_rate=self.exchanges_accepted
            / max(1, self.exchange_attempts),
            replica_states=self.replica_states,
            state_replicas=self.state_replicas,
            exchange_history=self.exchange_history,
            trajectory_files=[str(f) for f in self.trajectory_files],
            acceptance_matrix=self.acceptance_matrix,
            replica_visitation_histogram=self.replica_visit_counts,
            frames_per_replica=self._compute_frames_per_replica(),
            effective_sample_size=None,
        )
        with open(results_file, "wb") as pkl_f:
            pickle.dump({"remd": result}, pkl_f)
        with open(json_file, "w") as json_f:
            json.dump({"remd": result.to_dict(metadata_only=True)}, json_f)
        logger.info(f"Results saved to {results_file}")

    def tune_temperature_ladder(
        self, target_acceptance: Optional[float] = None
    ) -> List[float]:
        """Adjust the temperature ladder for a desired acceptance rate.

        Uses :func:`retune_temperature_ladder` to estimate new temperature
        spacings based on the accumulated pairwise acceptance statistics. The
        suggested temperatures replace ``self.temperatures``; callers should
        re-run :meth:`setup_replicas` before continuing the simulation.

        Parameters
        ----------
        target_acceptance:
            Desired neighbour acceptance probability. If ``None``,
            :attr:`target_accept` is used.

        Returns
        -------
        List[float]
            The updated temperature ladder.
        """

        target = (
            float(target_acceptance)
            if target_acceptance is not None
            else float(self.target_accept)
        )
        logger.info("Retuning temperature ladder toward target acceptance %.2f", target)
        stats = retune_temperature_ladder(
            self.temperatures,
            self.pair_attempt_counts,
            self.pair_accept_counts,
            target_acceptance=target,
            output_json=str(self.output_dir / "temperatures_suggested.json"),
            dry_run=True,
        )
        self.temperatures = stats["suggested_temperatures"]
        self.n_replicas = len(self.temperatures)
        return self.temperatures

    def _compute_frames_per_replica(self) -> List[int]:
        frames: List[int] = []
        for traj_file in self.trajectory_files:
            try:
                if traj_file.exists():
                    import mdtraj as md  # type: ignore

                    t = md.load(str(traj_file), top=self.pdb_file)
                    frames.append(int(t.n_frames))
                else:
                    frames.append(0)
            except Exception:
                frames.append(0)
        return frames

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

        # Reporter stride: prefer per-replica recorded stride, otherwise use planned stride
        default_stride = int(
            self.reporter_stride
            if self.reporter_stride is not None
            else max(1, self.dcd_stride)
        )

        # Load all trajectories and perform segment-wise demultiplexing
        demux_segments: List[Any] = []
        trajectory_frame_counts: Dict[str, int] = {}
        repaired_segments: List[int] = []

        n_segments = len(self.exchange_history)
        logger.info(f"Processing {n_segments} exchange steps (segments)...")
        logger.info(
            (
                f"Exchange frequency: {self.exchange_frequency} MD steps, "
                f"default DCD stride: {default_stride} MD steps"
            )
        )

        # Diagnostics for DCD files
        logger.info("DCD File Diagnostics:")
        loaded_trajs: Dict[int, Any] = {}
        for i, traj_file in enumerate(self.trajectory_files):
            if traj_file.exists():
                file_size = traj_file.stat().st_size
                logger.info(
                    f"  Replica {i}: {traj_file.name} exists, size: {file_size:,} bytes"
                )
                try:
                    import mdtraj as md  # type: ignore

                    t = md.load(str(traj_file), top=self.pdb_file)
                    loaded_trajs[i] = t
                    trajectory_frame_counts[str(traj_file)] = int(t.n_frames)
                    logger.info(f"    -> Loaded: {t.n_frames} frames")
                except Exception as e:
                    logger.warning(f"    -> Failed to load: {e}")
                    trajectory_frame_counts[str(traj_file)] = 0
            else:
                logger.warning(f"  Replica {i}: {traj_file.name} does not exist")

        if not loaded_trajs:
            logger.warning("No trajectories could be loaded for demultiplexing")
            return None

        # Effective equilibration steps actually integrated (heating + temp equil)
        if equilibration_steps > 0:
            effective_equil_steps = max(100, equilibration_steps * 40 // 100) + max(
                100, equilibration_steps * 60 // 100
            )
        else:
            effective_equil_steps = 0

        # Prepare temperature schedule mapping
        temp_schedule: Dict[str, Dict[str, float]] = {
            str(rid): {} for rid in range(self.n_replicas)
        }

        frames_per_segment: Optional[int] = None
        expected_start_frame = 0
        prev_stop_md = effective_equil_steps

        # Build per-segment slices
        for s, replica_states in enumerate(self.exchange_history):
            try:
                # Record temperature assignment for provenance
                for replica_idx, temp_state in enumerate(replica_states):
                    temp_schedule[str(replica_idx)][str(s)] = float(
                        self.temperatures[int(temp_state)]
                    )

                # Which replica was at the target temperature during this segment
                replica_at_target = None
                for replica_idx, temp_state in enumerate(replica_states):
                    if temp_state == target_temp_idx:
                        replica_at_target = int(replica_idx)
                        break

                # Segment MD step range [start, stop)
                start_md = effective_equil_steps + s * self.exchange_frequency
                stop_md = effective_equil_steps + (s + 1) * self.exchange_frequency

                if start_md < prev_stop_md:
                    raise DemuxIntegrityError("Non-monotonic segment times detected")
                prev_stop_md = stop_md

                if replica_at_target is None:
                    # Missing swap; fill using nearest neighbour if possible
                    if demux_segments and frames_per_segment is not None:
                        import mdtraj as md  # type: ignore

                        fill = md.join(
                            [
                                demux_segments[-1][-1:]
                                for _ in range(int(frames_per_segment))
                            ]
                        )
                        demux_segments.append(fill)
                        repaired_segments.append(s)
                        expected_start_frame += int(frames_per_segment)
                        logger.warning(
                            f"Segment {s} missing target replica - filled with nearest neighbour frame"
                        )
                        continue
                    raise DemuxIntegrityError(
                        f"Segment {s} missing target replica and no data to fill"
                    )

                traj = loaded_trajs.get(replica_at_target)
                if traj is None:
                    if demux_segments and frames_per_segment is not None:
                        import mdtraj as md  # type: ignore

                        fill = md.join(
                            [
                                demux_segments[-1][-1:]
                                for _ in range(int(frames_per_segment))
                            ]
                        )
                        demux_segments.append(fill)
                        repaired_segments.append(s)
                        expected_start_frame += int(frames_per_segment)
                        logger.warning(
                            f"Segment {s} missing trajectory data - filled with nearest neighbour frame"
                        )
                        continue
                    raise DemuxIntegrityError(
                        f"Segment {s} missing trajectory data and no data to fill"
                    )

                # Map to saved frame indices using replica's recorded stride if available
                stride = (
                    self._replica_reporter_stride[replica_at_target]
                    if replica_at_target < len(self._replica_reporter_stride)
                    else default_stride
                )
                start_frame = max(0, start_md // stride)
                # Inclusive of frames with step < stop_md
                end_frame = min(traj.n_frames, (max(0, stop_md - 1) // stride) + 1)

                if start_frame > expected_start_frame:
                    if demux_segments:
                        import mdtraj as md  # type: ignore

                        gap = start_frame - expected_start_frame
                        fill = md.join(
                            [demux_segments[-1][-1:] for _ in range(int(gap))]
                        )
                        demux_segments.append(fill)
                        repaired_segments.append(s)
                        expected_start_frame = start_frame
                        logger.warning(
                            f"Filled {gap} missing frame(s) before segment {s}"
                        )
                    else:
                        # Tolerate initial offset: start demux at the first available frame
                        gap = start_frame - expected_start_frame
                        expected_start_frame = start_frame
                        logger.warning(
                            f"Initial gap of {gap} frame(s) before first segment; starting at first available frame"
                        )
                elif start_frame < expected_start_frame:
                    raise DemuxIntegrityError("Non-monotonic frame indices detected")

                if end_frame > start_frame:
                    segment = traj[start_frame:end_frame]
                    demux_segments.append(segment)
                    if frames_per_segment is None:
                        frames_per_segment = int(end_frame - start_frame)
                    expected_start_frame = end_frame
                else:
                    if demux_segments and frames_per_segment is not None:
                        import mdtraj as md  # type: ignore

                        fill = md.join(
                            [
                                demux_segments[-1][-1:]
                                for _ in range(int(frames_per_segment))
                            ]
                        )
                        demux_segments.append(fill)
                        repaired_segments.append(s)
                        expected_start_frame += int(frames_per_segment)
                        logger.warning(
                            f"Segment {s} has no frames - filled with nearest neighbour frame"
                        )
                    else:
                        raise DemuxIntegrityError(
                            f"No frames available for segment {s}"
                        )
            except DemuxIntegrityError:
                raise
            except Exception:
                continue

        if demux_segments:
            try:
                import mdtraj as md  # type: ignore

                demux_traj = md.join(demux_segments)
                demux_file = self.output_dir / f"demux_T{actual_temp:.0f}K.dcd"
                demux_traj.save_dcd(str(demux_file))
                logger.info(f"Demultiplexed trajectory saved: {demux_file}")
                logger.info(
                    f"Total frames at target temperature: {int(demux_traj.n_frames)}"
                )

                timestep_ps = float(
                    self.integrators[0].getStepSize().value_in_unit(unit.picoseconds)
                    if self.integrators
                    else 0.0
                )
                metadata = DemuxMetadata(
                    exchange_frequency_steps=int(self.exchange_frequency),
                    integration_timestep_ps=timestep_ps,
                    frames_per_segment=int(frames_per_segment or 0),
                    temperature_schedule=temp_schedule,
                )
                meta_path = demux_file.with_suffix(".meta.json")
                metadata.to_json(meta_path)
                logger.info(f"Demultiplexed metadata saved: {meta_path}")

                if repaired_segments:
                    logger.warning(f"Repaired segments: {repaired_segments}")

                return str(demux_file)
            except Exception as e:
                logger.error(f"Error saving demultiplexed trajectory: {e}")
                return None
        else:
            logger.warning(
                (
                    "No segments found for demultiplexing - check exchange history, "
                    "frame indexing, or stride settings"
                )
            )
            logger.debug(f"  Exchange steps: {len(self.exchange_history)}")
            logger.debug(f"  Exchange frequency: {self.exchange_frequency}")
            logger.debug(f"  Effective equilibration steps: {effective_equil_steps}")
            logger.debug(f"  Default DCD stride: {default_stride}")
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

        # Normalize to get probabilities (not currently used downstream)
        _ = replica_visits / max(1, len(self.exchange_history))

        extra = compute_exchange_statistics(
            self.exchange_history,
            self.n_replicas,
            self.pair_attempt_counts,
            self.pair_accept_counts,
        )

        return {
            "total_exchange_attempts": self.exchange_attempts,
            "total_exchanges_accepted": self.exchanges_accepted,
            "overall_acceptance_rate": self.exchanges_accepted
            / max(1, self.exchange_attempts),
            **extra,
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
    target_accept: float = 0.30,
    tuning_steps: int = 0,
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
        target_accept: Desired per-pair acceptance probability when tuning
        tuning_steps: If >0, run a short pre-production simulation for ladder
            tuning with this many steps

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
        target_accept=target_accept,
    )

    # Plan DCD reporter stride before reporters are created in setup
    remd.plan_reporter_stride(
        total_steps=total_steps,
        equilibration_steps=equilibration_steps,
        target_frames=5000,
    )

    # Set up replicas
    remd.setup_replicas(bias_variables=bias_variables)

    if tuning_steps > 0:
        remd.run_simulation(total_steps=tuning_steps, equilibration_steps=0)
        remd.tune_temperature_ladder()
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
            target_accept=target_accept,
        )

        # Set up bias variables if they were used
        bias_variables = remd_config.get("bias_variables") if use_metadynamics else None

        # Plan reporter stride prior to setup
        remd.plan_reporter_stride(
            total_steps=total_steps,
            equilibration_steps=equilibration_steps,
            target_frames=5000,
        )

        # Only setup replicas if we haven't done energy minimization yet
        if not checkpoint_manager.is_step_completed("energy_minimization"):
            remd.setup_replicas(bias_variables=bias_variables)
            if tuning_steps > 0:
                remd.run_simulation(total_steps=tuning_steps, equilibration_steps=0)
                remd.tune_temperature_ladder()
                remd.setup_replicas(bias_variables=bias_variables)
    else:
        # Non-checkpoint mode (legacy)
        bias_variables = setup_bias_variables(pdb_file) if use_metadynamics else None
        remd = ReplicaExchange(
            pdb_file=pdb_file,
            temperatures=temperatures,
            output_dir=output_dir,
            exchange_frequency=50,
            target_accept=target_accept,
        )
        # Plan reporter stride prior to setup
        remd.plan_reporter_stride(
            total_steps=total_steps,
            equilibration_steps=equilibration_steps,
            target_frames=5000,
        )
        remd.setup_replicas(bias_variables=bias_variables)
        if tuning_steps > 0:
            remd.run_simulation(total_steps=tuning_steps, equilibration_steps=0)
            remd.tune_temperature_ladder()
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
                logger.info(f"Demultiplexing successful: {demux_traj}")
                if checkpoint_manager:
                    checkpoint_manager.mark_step_completed(
                        "trajectory_demux", {"demux_file": demux_traj}
                    )
            else:
                logger.warning("Demultiplexing returned no trajectory")
                if checkpoint_manager:
                    checkpoint_manager.mark_step_failed(
                        "trajectory_demux", "No frames found for demultiplexing"
                    )
        except Exception as e:
            logger.warning(f"Demultiplexing failed: {e}")
            if checkpoint_manager:
                checkpoint_manager.mark_step_failed("trajectory_demux", str(e))

        # Always log that the simulation itself was successful
        logger.info("REMD simulation completed successfully")
        logger.info("Raw trajectory files are available for manual analysis")
    else:
        logger.info(
            "Trajectory demux already completed or checkpoint manager not available"
        )

    # Print statistics
    stats = remd.get_exchange_statistics()
    logger.info(f"REMD Statistics: {stats}")

    return demux_traj
