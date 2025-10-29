# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Replica setup and initialization for REMD simulations.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import openmm
from openmm import Platform, unit
from openmm.app import PDBFile, Simulation

from pmarlo import constants as const
from pmarlo.features.deeptica.export import load_cv_model_info
from pmarlo.utils.path_utils import ensure_directory
from pmarlo.utils.validation import all_finite

from .platform_selector import select_platform_and_properties
from .running_stats import RunningStats
from .system_builder import (
    create_system,
    load_pdb_and_forcefield,
    log_system_info,
    setup_metadynamics,
)
from .trajectory import ClosableDCDReporter

logger = logging.getLogger("pmarlo")


class ReplicaSetup:
    """Handles replica initialization and setup operations."""

    @staticmethod
    def create_integrator_for_temperature(
        temperature: float, random_seed: int
    ) -> openmm.Integrator:
        """Create a Langevin integrator for the given temperature."""
        from ..utils.integrator import create_langevin_integrator

        return create_langevin_integrator(temperature, random_seed)

    @staticmethod
    def create_simulation(
        pdb: PDBFile,
        system: openmm.System,
        integrator: openmm.Integrator,
        platform: Platform,
        platform_properties: dict,
    ) -> Simulation:
        """Create an OpenMM simulation object."""
        return Simulation(
            pdb.topology, system, integrator, platform, platform_properties or None
        )

    @staticmethod
    def reuse_minimized_positions_quick_minimize(
        simulation: Simulation,
        shared_minimized_positions,
        replica_index: int,
    ) -> bool:
        """Try to reuse minimized positions with quick touch-up."""
        try:
            simulation.context.setPositions(shared_minimized_positions)
            simulation.minimizeEnergy(
                maxIterations=50,
                tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
            )
            logger.info(
                f"  Reused minimized coordinates for replica {replica_index} "
                f"(quick touch-up)"
            )
            return True
        except Exception as exc:
            logger.warning(
                f"  Failed to reuse minimized coords for replica "
                f"{replica_index}: {exc}; falling back to full minimization"
            )
            return False

    @staticmethod
    def check_initial_energy(simulation: Simulation, replica_index: int) -> None:
        """Check and log initial energy."""
        try:
            initial_state = simulation.context.getState(getEnergy=True)
            initial_energy = initial_state.getPotentialEnergy()
            logger.info(
                f"  Initial energy for replica {replica_index}: {initial_energy}"
            )
            energy_val = initial_energy.value_in_unit(unit.kilojoules_per_mole)
            if abs(energy_val) > const.NUMERIC_HARD_ENERGY_LIMIT:
                logger.warning(
                    f"  Very high initial energy ({energy_val:.2e} kJ/mol) "
                    f"detected for replica {replica_index}"
                )
        except Exception as exc:
            logger.warning(
                f"  Could not check initial energy for replica {replica_index}: {exc}"
            )

    @staticmethod
    def perform_stage1_minimization(
        simulation: Simulation, replica_index: int
    ) -> bool:
        """Perform stage 1 minimization with multiple attempts."""
        minimization_success = False
        schedule = [(50, 100.0), (100, 50.0), (200, 10.0)]
        for attempt, (max_iter, tolerance_val) in enumerate(schedule):
            try:
                tolerance = tolerance_val * unit.kilojoules_per_mole / unit.nanometer
                simulation.minimizeEnergy(maxIterations=max_iter, tolerance=tolerance)
                logger.info(
                    f"  Stage 1 minimization completed for replica "
                    f"{replica_index} (attempt {attempt + 1})"
                )
                minimization_success = True
                break
            except Exception as exc:
                logger.warning(
                    f"  Stage 1 minimization attempt {attempt + 1} failed "
                    f"for replica {replica_index}: {exc}"
                )
                if attempt == len(schedule) - 1:
                    logger.error(
                        f"  All minimization attempts failed for replica {replica_index}"
                    )
                    raise RuntimeError(
                        f"Energy minimization failed for replica {replica_index} "
                        "after 3 attempts. Structure may be too distorted. "
                        "Consider: 1) Better initial structure, 2) Different "
                        "forcefield, 3) Manual structure preparation"
                    )
        return minimization_success

    @staticmethod
    def perform_stage2_minimization_and_validation(
        simulation: Simulation,
        replica_index: int,
        shared_minimized_positions,
    ):
        """Perform stage 2 minimization and validate results."""
        try:
            ReplicaSetup._stage2_minimize(simulation, replica_index)
            state = ReplicaSetup._get_state_with_positions(simulation)
            energy = state.getPotentialEnergy()
            positions = state.getPositions()
            ReplicaSetup._validate_energy(energy, replica_index)
            ReplicaSetup._validate_positions(positions, replica_index)
            logger.info(f"  Final energy for replica {replica_index}: {energy}")
            if shared_minimized_positions is None:
                shared_minimized_positions = ReplicaSetup._cache_minimized_positions_safe(
                    state
                )
            return shared_minimized_positions
        except Exception as exc:
            ReplicaSetup._log_stage2_failure(replica_index, exc)
            ReplicaSetup._log_using_stage1_energy(simulation, replica_index)
            return shared_minimized_positions

    @staticmethod
    def _stage2_minimize(simulation: Simulation, replica_index: int) -> None:
        """Perform stage 2 minimization."""
        simulation.minimizeEnergy(
            maxIterations=100, tolerance=1.0 * unit.kilojoules_per_mole / unit.nanometer
        )
        logger.info(f"  Stage 2 minimization completed for replica {replica_index}")

    @staticmethod
    def _get_state_with_positions(simulation: Simulation):
        """Get simulation state with positions, energy, and velocities."""
        return simulation.context.getState(
            getPositions=True, getEnergy=True, getVelocities=True
        )

    @staticmethod
    def _validate_energy(energy, replica_index: int) -> None:
        """Validate energy values."""
        energy_val = float(energy.value_in_unit(unit.kilojoules_per_mole))
        if not all_finite(energy_val):
            raise ValueError(
                f"Invalid energy ({energy}) detected after minimization "
                f"for replica {replica_index}"
            )
        if abs(energy_val) > const.NUMERIC_SOFT_ENERGY_LIMIT:
            logger.warning(
                f"  High final energy ({energy_val:.2e} kJ/mol) for "
                f"replica {replica_index}"
            )

    @staticmethod
    def _validate_positions(positions, replica_index: int) -> None:
        """Validate position values."""
        pos_array = positions.value_in_unit(unit.nanometer)
        if not all_finite(pos_array):
            raise ValueError(
                f"Invalid positions detected after minimization for "
                f"replica {replica_index}"
            )

    @staticmethod
    def _cache_minimized_positions_safe(state):
        """Cache minimized positions safely."""
        try:
            logger.info("  Cached minimized coordinates from replica 0 for reuse")
            return state.getPositions()
        except Exception:
            return None

    @staticmethod
    def _log_stage2_failure(replica_index: int, exc: Exception) -> None:
        """Log stage 2 failure."""
        logger.error(
            f"  Stage 2 minimization or validation failed for replica "
            f"{replica_index}: {exc}"
        )
        logger.warning(
            f"  Attempting to continue with Stage 1 result for replica {replica_index}"
        )

    @staticmethod
    def _log_using_stage1_energy(simulation: Simulation, replica_index: int) -> None:
        """Log that we're using stage 1 energy."""
        try:
            state = simulation.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy()
            logger.info(f"  Using Stage 1 energy for replica {replica_index}: {energy}")
        except Exception:
            raise RuntimeError(
                f"Complete minimization failure for replica {replica_index}"
            )

    @staticmethod
    def add_dcd_reporter(
        simulation: Simulation, replica_index: int, output_dir: Path, reporter_stride: int
    ) -> Path:
        """Add DCD reporter to simulation."""
        traj_file = output_dir / f"replica_{replica_index:02d}.dcd"
        stride = int(reporter_stride)
        dcd_reporter = ClosableDCDReporter(str(traj_file), stride)
        simulation.reporters.append(dcd_reporter)
        return traj_file

    @staticmethod
    def initialize_cv_monitoring(cv_model_path: Optional[Path]):
        """Initialize CV monitoring if model path is provided."""
        if cv_model_path is None:
            return None, None, None

        try:
            import torch

            info = load_cv_model_info(cv_model_path.parent, cv_model_path.stem)
            cv_dim = int(info.get("config", {}).get("cv_dim", 0))
            if cv_dim <= 0:
                raise ValueError("cv_dim metadata missing from CV model config.")

            cv_monitor_module = torch.jit.load(str(cv_model_path), map_location="cpu")
            cv_monitor_module.eval()
            bias_energy_stats = RunningStats(dim=1)
            bias_cv_stats = RunningStats(dim=cv_dim)

            logger.info("CV monitoring initialized")
            return cv_monitor_module, bias_energy_stats, bias_cv_stats
        except Exception as exc:
            logger.warning(
                "Unable to initialise CV monitoring for logging: %s", exc
            )
            return None, None, None

    @staticmethod
    def setup_replicas(
        pdb_file: str,
        forcefield_files: List[str],
        temperatures: List[float],
        output_dir: Path,
        reporter_stride: int,
        random_seed: int,
        resume_pdb: Optional[Path],
        resume_jitter_sigma_nm: float,
        reseed_velocities: bool,
        cv_model_path: Optional[Path],
        cv_scaler_mean,
        cv_scaler_scale,
        bias_variables: Optional[List],
    ):
        """
        Set up all replica simulations with different temperatures.

        Returns:
            Tuple of (replicas, contexts, integrators, trajectory_files,
                     replica_reporter_stride, metadynamics, cv_monitor_data)
        """
        logger.info("Setting up replica simulations...")

        # Enforce stride planning before creating reporters
        assert (
            reporter_stride is not None and reporter_stride > 0
        ), "reporter_stride is not planned. Call plan_reporter_stride(...) before setup_replicas()"

        pdb, forcefield = load_pdb_and_forcefield(pdb_file, forcefield_files)
        resume_positions = ReplicaSetup._load_resume_positions(
            resume_pdb, resume_jitter_sigma_nm
        )

        cv_model_path_str = (
            str(cv_model_path) if cv_model_path is not None else None
        )

        system = create_system(
            pdb,
            forcefield,
            cv_model_path=cv_model_path_str,
            cv_scaler_mean=cv_scaler_mean,
            cv_scaler_scale=cv_scaler_scale,
        )
        log_system_info(system, logger)
        metadynamics = setup_metadynamics(
            system, bias_variables, temperatures[0], output_dir
        )

        # Initialize CV monitoring
        cv_monitor_module, bias_energy_stats, bias_cv_stats = (
            ReplicaSetup.initialize_cv_monitoring(cv_model_path)
        )

        platform, platform_properties = select_platform_and_properties(
            logger, prefer_deterministic=True if random_seed is not None else False
        )

        replicas = []
        contexts = []
        integrators = []
        trajectory_files = []
        replica_reporter_strides = []
        shared_minimized_positions = None

        for i, temperature in enumerate(temperatures):
            logger.info(f"Setting up replica {i} at {temperature}K...")

            integrator = ReplicaSetup.create_integrator_for_temperature(
                temperature, random_seed + i
            )
            simulation = ReplicaSetup.create_simulation(
                pdb, system, integrator, platform, platform_properties
            )

            # Set positions
            ReplicaSetup._set_initial_positions(
                simulation, resume_positions, pdb, reseed_velocities, temperature
            )

            # Try to reuse minimized positions
            if (
                shared_minimized_positions is not None
                and ReplicaSetup.reuse_minimized_positions_quick_minimize(
                    simulation, shared_minimized_positions, i
                )
            ):
                traj_file = ReplicaSetup.add_dcd_reporter(
                    simulation, i, output_dir, reporter_stride
                )
                replicas.append(simulation)
                integrators.append(integrator)
                contexts.append(simulation.context)
                trajectory_files.append(traj_file)
                replica_reporter_strides.append(reporter_stride)
                logger.info(f"Replica {i:02d}: T = {temperature:.1f} K")
                continue

            # Perform minimization
            logger.info(f"  Minimizing energy for replica {i}...")
            ReplicaSetup.check_initial_energy(simulation, i)
            minimization_success = ReplicaSetup.perform_stage1_minimization(
                simulation, i
            )

            if minimization_success:
                shared_minimized_positions = (
                    ReplicaSetup.perform_stage2_minimization_and_validation(
                        simulation, i, shared_minimized_positions
                    )
                )

            traj_file = ReplicaSetup.add_dcd_reporter(
                simulation, i, output_dir, reporter_stride
            )
            replicas.append(simulation)
            integrators.append(integrator)
            contexts.append(simulation.context)
            trajectory_files.append(traj_file)
            replica_reporter_strides.append(reporter_stride)
            logger.info(f"Replica {i:02d}: T = {temperature:.1f} K")

        logger.info("All replicas set up successfully")

        cv_monitor_data = {
            "module": cv_monitor_module,
            "energy_stats": bias_energy_stats,
            "cv_stats": bias_cv_stats,
        }

        return (
            replicas,
            contexts,
            integrators,
            trajectory_files,
            replica_reporter_strides,
            metadynamics,
            cv_monitor_data,
        )

    @staticmethod
    def _load_resume_positions(resume_pdb: Optional[Path], jitter_sigma_nm: float):
        """Load resume positions from PDB if available."""
        if resume_pdb is None or not resume_pdb.exists():
            return None

        try:
            pdb_resume = PDBFile(str(resume_pdb))
            # Optional small Gaussian jitter in nm
            if jitter_sigma_nm > 0.0:
                arr = np.array(
                    [[v.x, v.y, v.z] for v in pdb_resume.positions], dtype=float
                )
                noise = np.random.normal(
                    loc=0.0, scale=float(jitter_sigma_nm), size=arr.shape
                )
                arr = arr + noise
                from openmm import Vec3

                resume_positions = [
                    Vec3(float(x), float(y), float(z)) * unit.nanometer
                    for x, y, z in arr
                ]
            else:
                resume_positions = pdb_resume.positions
            logger.info(
                "Resuming replicas from PDB positions: %s (jitter_nm=%.4f)",
                str(resume_pdb),
                float(jitter_sigma_nm),
            )
            return resume_positions
        except Exception as exc:
            logger.warning("Failed to load resume PDB %s: %s", str(resume_pdb), exc)
            return None

    @staticmethod
    def _set_initial_positions(
        simulation: Simulation,
        resume_positions,
        pdb: PDBFile,
        reseed_velocities: bool,
        temperature: float,
    ) -> None:
        """Set initial positions and optionally reseed velocities."""
        try:
            if resume_positions is not None:
                simulation.context.setPositions(resume_positions)
            else:
                simulation.context.setPositions(pdb.positions)
        except Exception:
            simulation.context.setPositions(pdb.positions)

        # Optional velocity reseed on start
        if reseed_velocities:
            try:
                simulation.context.setVelocitiesToTemperature(
                    temperature * unit.kelvin
                )
            except Exception:
                pass

    @staticmethod
    def validate_temperature_ladder(temps: List[float]) -> None:
        """Validate that temperature ladder is properly configured."""
        if temps is None:
            raise ValueError("Temperature ladder is None")
        if len(temps) < 2:
            raise ValueError("Temperature ladder must have at least 2 values")
        last = None
        for t in temps:
            if float(t) <= 0.0:
                raise ValueError("Temperatures must be > 0 K")
            if last is not None and float(t) <= float(last):
                raise ValueError("Temperature ladder must be strictly increasing")
            last = t

