"""
Replica Exchange Molecular Dynamics (REMD) implementation for enhanced sampling.

This module provides functionality to run replica exchange simulations using OpenMM,
allowing for better exploration of conformational space across multiple temperatures.
"""

import numpy as np
import openmm
from openmm import unit
from openmm.app import *
from openmm import Platform
import mdtraj as md
from pathlib import Path
import os
import pickle
from typing import List, Tuple, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplicaExchange:
    """
    Replica Exchange Molecular Dynamics implementation using OpenMM.
    
    This class handles the setup and execution of REMD simulations,
    managing multiple temperature replicas and exchange attempts.
    """
    
    def __init__(self, 
                 pdb_file: str,
                 forcefield_files: List[str] = None,
                 temperatures: List[float] = None,
                 output_dir: str = "remd_output",
                 exchange_frequency: int = 50):  # Very frequent exchanges for testing
        """
        Initialize the replica exchange simulation.
        
        Args:
            pdb_file: Path to the prepared PDB file
            forcefield_files: List of forcefield XML files
            temperatures: List of temperatures in Kelvin for replicas
            output_dir: Directory to store output files
            exchange_frequency: Number of steps between exchange attempts
        """
        self.pdb_file = pdb_file
        self.forcefield_files = forcefield_files or ["amber14-all.xml", "amber14/tip3pfb.xml"]
        self.temperatures = temperatures or self._generate_temperature_ladder()
        self.output_dir = Path(output_dir)
        self.exchange_frequency = exchange_frequency
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize replicas
        self.n_replicas = len(self.temperatures)
        self.replicas = []
        self.contexts = []
        self.integrators = []
        
        # Exchange statistics
        self.exchange_attempts = 0
        self.exchanges_accepted = 0
        self.replica_states = list(range(self.n_replicas))  # Which temperature each replica is at
        self.state_replicas = list(range(self.n_replicas))  # Which replica is at each temperature
        
        # Simulation data
        self.trajectory_files = []
        self.energies = []
        self.exchange_history = []
        
        logger.info(f"Initialized REMD with {self.n_replicas} replicas")
        logger.info(f"Temperature range: {min(self.temperatures):.1f} - {max(self.temperatures):.1f} K")
    
    def _generate_temperature_ladder(self, 
                                   min_temp: float = 300.0, 
                                   max_temp: float = 350.0,  # Smaller range for better overlap
                                   n_replicas: int = 3) -> List[float]:  # Even fewer replicas for testing
        """
        Generate an exponential temperature ladder for optimal exchange efficiency.
        
        Args:
            min_temp: Minimum temperature in Kelvin
            max_temp: Maximum temperature in Kelvin
            n_replicas: Number of temperature replicas
            
        Returns:
            List of temperatures in Kelvin
        """
        # Exponential spacing for better overlap
        temperatures = min_temp * (max_temp / min_temp) ** (np.arange(n_replicas) / (n_replicas - 1))
        return temperatures.tolist()
    
    def setup_replicas(self, bias_variables: List = None):
        """
        Set up all replica simulations with different temperatures.
        
        Args:
            bias_variables: Optional list of bias variables for metadynamics
        """
        logger.info("Setting up replica simulations...")
        
        # Load PDB and create forcefield
        pdb = PDBFile(self.pdb_file)
        forcefield = ForceField(*self.forcefield_files)
        
        # Create system (same for all replicas) with conservative settings
        logger.info("Creating molecular system with conservative parameters...")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=PME,
            constraints=HBonds,  # Constrain bonds involving hydrogen
            rigidWater=True,     # Keep water molecules rigid
            nonbondedCutoff=1.0*unit.nanometer,
            ewaldErrorTolerance=5e-4,  # More conservative Ewald tolerance
            hydrogenMass=1.5*unit.amu,  # Slightly increase hydrogen mass for stability
            removeCMMotion=True  # Remove center-of-mass motion
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
                temperature=self.temperatures[0] * unit.kelvin,  # Will be updated for each replica
                biasFactor=10.0,
                height=1.0 * unit.kilojoules_per_mole,
                frequency=500,
                biasDir=str(bias_dir),
                saveFrequency=1000
            )
        
        # Create replicas with different temperatures
        # Use Reference platform for stability - it's slower but more robust
        try:
            platform = Platform.getPlatformByName("Reference")
            logger.info("Using Reference platform for stability")
        except:
            try:
                platform = Platform.getPlatformByName("CPU")
                logger.info("Using CPU platform")
            except:
                platform = Platform.getPlatformByName("CUDA")
                logger.info("Using CUDA platform")
        
        for i, temperature in enumerate(self.temperatures):
            logger.info(f"Setting up replica {i} at {temperature}K...")
            
            # Create integrator for this temperature with conservative timestep
            integrator = openmm.LangevinIntegrator(
                temperature * unit.kelvin,
                1.0 / unit.picosecond,
                1.0 * unit.femtoseconds  # Reduced from 2.0 fs for stability
            )
            
            # Create simulation
            simulation = Simulation(pdb.topology, system, integrator, platform)
            simulation.context.setPositions(pdb.positions)
            
            # Multi-stage energy minimization for stability
            logger.info(f"  Minimizing energy for replica {i}...")
            
            # Stage 1: Initial minimization with steepest descent
            try:
                simulation.minimizeEnergy(maxIterations=100, tolerance=10.0*unit.kilojoules_per_mole/unit.nanometer)
                logger.info(f"  Stage 1 minimization completed for replica {i}")
            except Exception as e:
                logger.error(f"  Stage 1 minimization failed for replica {i}: {e}")
                raise
            
            # Stage 2: Refined minimization  
            try:
                simulation.minimizeEnergy(maxIterations=100, tolerance=1.0*unit.kilojoules_per_mole/unit.nanometer)
                logger.info(f"  Stage 2 minimization completed for replica {i}")
                
                # Check for NaN after minimization
                state = simulation.context.getState(getPositions=True, getEnergy=True)
                energy = state.getPotentialEnergy()
                if str(energy) == 'nan' or 'nan' in str(energy).lower():
                    raise ValueError(f"NaN energy detected after minimization for replica {i}")
                    
                logger.info(f"  Final energy for replica {i}: {energy}")
                
            except Exception as e:
                logger.error(f"  Stage 2 minimization failed for replica {i}: {e}")
                raise
            
            # Set up trajectory reporter
            traj_file = self.output_dir / f"replica_{i:02d}.dcd"
            dcd_reporter = DCDReporter(str(traj_file), 10)  # Save every 10 steps
            simulation.reporters.append(dcd_reporter)
            
            # Store replica data
            self.replicas.append(simulation)
            self.integrators.append(integrator)
            self.contexts.append(simulation.context)
            self.trajectory_files.append(traj_file)
            
            logger.info(f"Replica {i:02d}: T = {temperature:.1f} K")
        
        logger.info("All replicas set up successfully")
    
    def calculate_exchange_probability(self, replica_i: int, replica_j: int) -> float:
        """
        Calculate the probability of exchanging two replicas.
        
        Args:
            replica_i: Index of first replica
            replica_j: Index of second replica
            
        Returns:
            Exchange probability
        """
        # Get current energies
        state_i = self.contexts[replica_i].getState(getEnergy=True)
        state_j = self.contexts[replica_j].getState(getEnergy=True)
        
        energy_i = state_i.getPotentialEnergy()
        energy_j = state_j.getPotentialEnergy()
        
        # Get temperatures
        temp_i = self.temperatures[self.replica_states[replica_i]]
        temp_j = self.temperatures[self.replica_states[replica_j]]
        
        # Calculate exchange probability using Metropolis criterion
        # Use molar gas constant since energies are per mole
        RT_i = unit.MOLAR_GAS_CONSTANT_R * temp_i * unit.kelvin
        RT_j = unit.MOLAR_GAS_CONSTANT_R * temp_j * unit.kelvin
        
        # Calculate each term separately to ensure proper unit handling
        # OpenMM sometimes returns float instead of Quantity for dimensionless results
        def safe_dimensionless(quantity):
            """Safely extract dimensionless value from OpenMM quantity or float."""
            if hasattr(quantity, 'value_in_unit'):
                return quantity.value_in_unit(unit.dimensionless)
            else:
                # Already a dimensionless float
                return float(quantity)
        
        term1 = safe_dimensionless(energy_i / RT_j)
        term2 = safe_dimensionless(energy_j / RT_i)
        term3 = safe_dimensionless(energy_i / RT_i)
        term4 = safe_dimensionless(energy_j / RT_j)
        
        delta = (term1 + term2) - (term3 + term4)
        prob = min(1.0, np.exp(-delta))
        
        # Debug logging for troubleshooting low acceptance rates
        logger.info(f"Exchange calculation: E_i={energy_i}, E_j={energy_j}, T_i={temp_i:.1f}K, T_j={temp_j:.1f}K, delta={delta:.3f}, prob={prob:.6f}")
        
        return prob
    
    def attempt_exchange(self, replica_i: int, replica_j: int) -> bool:
        """
        Attempt to exchange two replicas.
        
        Args:
            replica_i: Index of first replica
            replica_j: Index of second replica
            
        Returns:
            True if exchange was accepted, False otherwise
        """
        self.exchange_attempts += 1
        
        # Calculate exchange probability
        prob = self.calculate_exchange_probability(replica_i, replica_j)
        
        # Accept or reject exchange
        if np.random.random() < prob:
            # Perform exchange by swapping temperatures
            # Save current states
            state_i = self.contexts[replica_i].getState(getPositions=True, getVelocities=True)
            state_j = self.contexts[replica_j].getState(getPositions=True, getVelocities=True)
            
            # Update temperature mappings
            old_state_i = self.replica_states[replica_i]
            old_state_j = self.replica_states[replica_j]
            
            self.replica_states[replica_i] = old_state_j
            self.replica_states[replica_j] = old_state_i
            
            self.state_replicas[old_state_i] = replica_j
            self.state_replicas[old_state_j] = replica_i
            
            # Update integrator temperatures
            self.integrators[replica_i].setTemperature(self.temperatures[old_state_j] * unit.kelvin)
            self.integrators[replica_j].setTemperature(self.temperatures[old_state_i] * unit.kelvin)
            
            # Set swapped states
            self.contexts[replica_i].setState(state_j)
            self.contexts[replica_j].setState(state_i)
            
            self.exchanges_accepted += 1
            
            logger.debug(f"Exchange accepted: replica {replica_i} <-> {replica_j} (prob={prob:.3f})")
            return True
        else:
            logger.debug(f"Exchange rejected: replica {replica_i} <-> {replica_j} (prob={prob:.3f})")
            return False
    
    def run_simulation(self, 
                      total_steps: int = 1000,  # Very fast for testing
                      equilibration_steps: int = 100,  # Minimal equilibration
                      save_state_frequency: int = 1000,
                      checkpoint_manager=None):
        """
        Run the replica exchange simulation.
        
        Args:
            total_steps: Total number of MD steps to run
            equilibration_steps: Number of equilibration steps before data collection
            save_state_frequency: Frequency to save simulation states
            checkpoint_manager: CheckpointManager instance for state tracking
        """
        logger.info(f"Starting REMD simulation: {total_steps} steps")
        logger.info(f"Exchange attempts every {self.exchange_frequency} steps")
        
        # Gradual heating and equilibration phase
        if equilibration_steps > 0:
            # Check if gradual heating is already completed
            if checkpoint_manager and checkpoint_manager.is_step_completed("gradual_heating"):
                logger.info("Gradual heating already completed ✓")
            else:
                if checkpoint_manager:
                    checkpoint_manager.mark_step_started("gradual_heating")
                logger.info(f"Equilibration with gradual heating: {equilibration_steps} steps")
            
                # Phase 1: Gradual heating (first 40% of equilibration)
                heating_steps = max(100, equilibration_steps * 40 // 100)
                logger.info(f"   Phase 1: Gradual heating over {heating_steps} steps")
                
                heating_chunk_size = max(10, heating_steps // 20)  # Heat in 20 stages
                for heat_step in range(0, heating_steps, heating_chunk_size):
                    current_steps = min(heating_chunk_size, heating_steps - heat_step)
                    
                    # Calculate gradual temperature scaling (start from 50K, ramp to target)
                    progress_fraction = (heat_step + current_steps) / heating_steps
                    
                    for replica_idx, replica in enumerate(self.replicas):
                        target_temp = self.temperatures[self.replica_states[replica_idx]]
                        current_temp = 50.0 + (target_temp - 50.0) * progress_fraction
                        
                        # Update integrator temperature gradually
                        replica.integrator.setTemperature(current_temp * unit.kelvin)
                        
                        # Run with error recovery
                        try:
                            replica.step(current_steps)
                        except Exception as e:
                            if "NaN" in str(e) or "nan" in str(e).lower():
                                logger.warning(f"   NaN detected in replica {replica_idx} during heating, attempting recovery...")
                                
                                # Attempt recovery by resetting velocities and reducing step size
                                replica.context.setVelocitiesToTemperature(current_temp * unit.kelvin)
                                
                                # Try smaller steps
                                small_steps = max(1, current_steps // 5)
                                for recovery_attempt in range(5):
                                    try:
                                        replica.step(small_steps)
                                        break
                                    except:
                                        if recovery_attempt == 4:  # Last attempt failed
                                            raise RuntimeError(f"Failed to recover from NaN in replica {replica_idx}")
                                        replica.context.setVelocitiesToTemperature(current_temp * unit.kelvin * 0.9)
                            else:
                                raise
                    
                    progress = min(40, (heat_step + current_steps) * 40 // heating_steps)
                    logger.info(f"   Heating Progress: {progress}% - Current temps: {[50.0 + (self.temperatures[self.replica_states[i]] - 50.0) * progress_fraction for i in range(len(self.replicas))]}")
                
                # Mark heating as completed
                if checkpoint_manager:
                    checkpoint_manager.mark_step_completed("gradual_heating", {
                        "heating_steps": heating_steps,
                        "final_temperatures": [self.temperatures[state] for state in self.replica_states]
                    })
            
            # Phase 2: Temperature equilibration (remaining 60% of equilibration)
            if checkpoint_manager and checkpoint_manager.is_step_completed("equilibration"):
                logger.info("Temperature equilibration already completed ✓")
            else:
                if checkpoint_manager:
                    checkpoint_manager.mark_step_started("equilibration")
                
                temp_equil_steps = max(100, equilibration_steps * 60 // 100)  # Calculate correctly
                logger.info(f"   Phase 2: Temperature equilibration at target temperatures over {temp_equil_steps} steps")
                
                # Set all replicas to their final target temperatures
                for replica_idx, replica in enumerate(self.replicas):
                    target_temp = self.temperatures[self.replica_states[replica_idx]]
                    replica.integrator.setTemperature(target_temp * unit.kelvin)
                    replica.context.setVelocitiesToTemperature(target_temp * unit.kelvin)
                
                equil_chunk_size = max(1, temp_equil_steps // 10)  # 10% chunks
                for i in range(0, temp_equil_steps, equil_chunk_size):
                    current_steps = min(equil_chunk_size, temp_equil_steps - i)
                    
                    for replica_idx, replica in enumerate(self.replicas):
                        try:
                            replica.step(current_steps)
                        except Exception as e:
                            if "NaN" in str(e) or "nan" in str(e).lower():
                                logger.error(f"   NaN detected in replica {replica_idx} during equilibration - simulation unstable")
                                if checkpoint_manager:
                                    checkpoint_manager.mark_step_failed("equilibration", str(e))
                                raise RuntimeError(f"Simulation became unstable for replica {replica_idx}. Try: 1) Better initial structure, 2) Smaller timestep, 3) More minimization")
                            else:
                                raise
                    
                    progress = min(100, 40 + (i + current_steps) * 60 // temp_equil_steps)
                    logger.info(f"   Equilibration Progress: {progress}% ({equilibration_steps - temp_equil_steps + i + current_steps}/{equilibration_steps} steps)")
                
                if checkpoint_manager:
                    checkpoint_manager.mark_step_completed("equilibration", {
                        "equilibration_steps": temp_equil_steps,
                        "total_equilibration": equilibration_steps
                    })
                
                logger.info("   Equilibration Complete ✓")
        
        # Production phase with exchanges
        if checkpoint_manager and checkpoint_manager.is_step_completed("production_simulation"):
            logger.info("Production simulation already completed ✓")
            return  # Skip production phase
        
        if checkpoint_manager:
            checkpoint_manager.mark_step_started("production_simulation")
        
        production_steps = total_steps - equilibration_steps
        exchange_steps = production_steps // self.exchange_frequency
        
        logger.info(f"Production: {production_steps} steps with {exchange_steps} exchange attempts")
        
        for step in range(exchange_steps):
            # Run MD for all replicas with error recovery
            for replica_idx, replica in enumerate(self.replicas):
                try:
                    replica.step(self.exchange_frequency)
                except Exception as e:
                    if "NaN" in str(e) or "nan" in str(e).lower():
                        logger.error(f"NaN detected in replica {replica_idx} during production phase")
                        # Try to save trajectory data before failing
                        try:
                            state = replica.context.getState(getPositions=True, getVelocities=True)
                            logger.info(f"Attempting to save current state before failure...")
                        except:
                            pass
                        raise RuntimeError(f"Simulation became unstable for replica {replica_idx} at production step {step}. "
                                         f"Consider: 1) Longer equilibration, 2) Smaller timestep, 3) Different initial structure")
                    else:
                        raise
            
            # Attempt exchanges between adjacent temperatures
            for i in range(0, self.n_replicas - 1, 2):  # Even pairs
                try:
                    self.attempt_exchange(i, i + 1)
                except Exception as e:
                    logger.warning(f"Exchange attempt failed between replicas {i} and {i+1}: {e}")
                    # Continue with other exchanges
            
            for i in range(1, self.n_replicas - 1, 2):  # Odd pairs
                self.attempt_exchange(i, i + 1)
            
            # Save exchange history
            self.exchange_history.append(self.replica_states.copy())
            
            # Enhanced progress reporting - show every step for fast runs
            progress_percent = (step + 1) * 100 // exchange_steps
            acceptance_rate = self.exchanges_accepted / max(1, self.exchange_attempts)
            completed_steps = (step + 1) * self.exchange_frequency + equilibration_steps
            
            logger.info(f"   Production Progress: {progress_percent}% "
                       f"({step + 1}/{exchange_steps} exchanges, "
                       f"{completed_steps}/{total_steps} total steps) "
                       f"| Acceptance: {acceptance_rate:.3f}")
            
            # Save states periodically
            if (step + 1) * self.exchange_frequency % save_state_frequency == 0:
                self.save_checkpoint(step + 1)
        
        # Mark production as completed
        if checkpoint_manager:
            checkpoint_manager.mark_step_completed("production_simulation", {
                "production_steps": production_steps,
                "exchange_steps": exchange_steps,
                "final_acceptance_rate": self.exchanges_accepted / max(1, self.exchange_attempts)
            })
        
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
            dcd_reporters = [r for r in replica.reporters if hasattr(r, '_out')]
            for reporter in dcd_reporters:
                try:
                    # Force close the DCD file
                    if hasattr(reporter, '_out') and reporter._out:
                        reporter._out.close()
                        logger.debug(f"Closed DCD file for replica {i}")
                except Exception as e:
                    logger.warning(f"Error closing DCD file for replica {i}: {e}")
            
            # Remove DCD reporters from the simulation
            replica.reporters = [r for r in replica.reporters if not hasattr(r, '_out')]
        
        # Force garbage collection to ensure file handles are released
        import gc
        gc.collect()
        
        logger.info("DCD files closed and flushed")
    
    def save_checkpoint(self, step: int):
        """Save simulation checkpoint."""
        checkpoint_file = self.output_dir / f"checkpoint_step_{step:06d}.pkl"
        checkpoint_data = {
            'step': step,
            'replica_states': self.replica_states,
            'state_replicas': self.state_replicas,
            'exchange_attempts': self.exchange_attempts,
            'exchanges_accepted': self.exchanges_accepted,
            'exchange_history': self.exchange_history
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def save_results(self):
        """Save final simulation results."""
        results_file = self.output_dir / "remd_results.pkl"
        results = {
            'temperatures': self.temperatures,
            'n_replicas': self.n_replicas,
            'exchange_frequency': self.exchange_frequency,
            'exchange_attempts': self.exchange_attempts,
            'exchanges_accepted': self.exchanges_accepted,
            'final_acceptance_rate': self.exchanges_accepted / max(1, self.exchange_attempts),
            'replica_states': self.replica_states,
            'state_replicas': self.state_replicas,
            'exchange_history': self.exchange_history,
            'trajectory_files': [str(f) for f in self.trajectory_files]
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {results_file}")
    
    def demux_trajectories(self, target_temperature: float = 300.0, equilibration_steps: int = 100) -> str:
        """
        Demultiplex trajectories to extract frames at target temperature.
        
        Args:
            target_temperature: Target temperature to extract frames for
            equilibration_steps: Number of equilibration steps (needed for frame calculation)
            
        Returns:
            Path to the demultiplexed trajectory file
        """
        logger.info(f"Demultiplexing trajectories for T = {target_temperature} K")
        
        # Find the target temperature index
        target_temp_idx = np.argmin(np.abs(np.array(self.temperatures) - target_temperature))
        actual_temp = self.temperatures[target_temp_idx]
        
        logger.info(f"Using closest temperature: {actual_temp:.1f} K")
        
        # Check if we have exchange history
        if not self.exchange_history:
            logger.warning("No exchange history available for demultiplexing")
            return None
        
        # DCD reporter settings (must match setup_replicas)
        dcd_frequency = 10  # Frames saved every 10 MD steps (from setup_replicas)
        
        # Load all trajectories with proper error handling
        demux_frames = []
        trajectory_frame_counts = {}  # Cache frame counts to avoid repeated loading
        
        logger.info(f"Processing {len(self.exchange_history)} exchange steps...")
        logger.info(f"Exchange frequency: {self.exchange_frequency} MD steps, DCD frequency: {dcd_frequency} MD steps")
        
        # Debug: Check files exist and get basic info
        logger.info("DCD File Diagnostics:")
        for i, traj_file in enumerate(self.trajectory_files):
            if traj_file.exists():
                file_size = traj_file.stat().st_size
                logger.info(f"  Replica {i}: {traj_file.name} exists, size: {file_size:,} bytes")
                
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
        
        for step, replica_states in enumerate(self.exchange_history):
            try:
                # Find which replica was at the target temperature at this step
                replica_at_target = None
                for replica_idx, temp_state in enumerate(replica_states):
                    if temp_state == target_temp_idx:
                        replica_at_target = replica_idx
                        break
                
                if replica_at_target is None:
                    logger.debug(f"No replica at target temperature {actual_temp}K at exchange step {step}")
                    continue
                
                # Calculate the correct frame number in the DCD file
                # Exchange step corresponds to: equilibration_steps + step * exchange_frequency MD steps
                md_step = equilibration_steps + step * self.exchange_frequency
                frame_number = md_step // dcd_frequency
                
                # Debug detailed frame calculation
                if step < 3:  # Log first few calculations
                    logger.info(f"Frame calculation debug - Exchange step {step}:")
                    logger.info(f"  Replica {replica_at_target} at target T={actual_temp}K")
                    logger.info(f"  MD step = {equilibration_steps} (equilibration) + {step} * {self.exchange_frequency} = {md_step}")
                    logger.info(f"  Frame = {md_step} // {dcd_frequency} = {frame_number}")
                else:
                    logger.debug(f"Exchange step {step}: Replica {replica_at_target} at target T={actual_temp}K, "
                               f"MD step {md_step}, frame {frame_number}")
                
                # Get the trajectory file for this replica
                traj_file = self.trajectory_files[replica_at_target]
                
                if not traj_file.exists():
                    logger.warning(f"Trajectory file not found: {traj_file}")
                    continue
                
                # Get frame count for this trajectory (with caching)
                if str(traj_file) not in trajectory_frame_counts:
                    try:
                        # Load with topology for DCD files
                        temp_traj = md.load(str(traj_file), top=self.pdb_file)
                        trajectory_frame_counts[str(traj_file)] = temp_traj.n_frames
                        logger.debug(f"Trajectory {traj_file.name} has {temp_traj.n_frames} frames")
                    except Exception as e:
                        logger.warning(f"Could not load trajectory {traj_file}: {e}")
                        trajectory_frame_counts[str(traj_file)] = 0
                        continue
                
                n_frames = trajectory_frame_counts[str(traj_file)]
                
                # Check if the requested frame exists
                if frame_number < n_frames:
                    try:
                        frame = md.load_frame(str(traj_file), frame_number, top=self.pdb_file)
                        demux_frames.append(frame)
                        logger.debug(f"Loaded frame {frame_number} from replica {replica_at_target} (T={actual_temp}K)")
                    except Exception as e:
                        logger.warning(f"Failed to load frame {frame_number} from {traj_file.name}: {e}")
                        continue
                else:
                    logger.debug(f"Frame {frame_number} not available in trajectory {traj_file.name} (has {n_frames} frames)")
                    
            except Exception as e:
                logger.warning(f"Error processing exchange step {step}: {e}")
                continue
        
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
            logger.warning("No frames found for demultiplexing - this may indicate frame indexing issues")
            logger.info("Debug info:")
            logger.info(f"  Exchange steps: {len(self.exchange_history)}")
            logger.info(f"  Exchange frequency: {self.exchange_frequency}")
            logger.info(f"  Equilibration steps: {equilibration_steps}")
            logger.info(f"  DCD frequency: {dcd_frequency}")
            for i, traj_file in enumerate(self.trajectory_files):
                n_frames = trajectory_frame_counts.get(str(traj_file), 0)
                logger.info(f"  Replica {i}: {n_frames} frames in {traj_file.name}")
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
        
        return {
            'total_exchange_attempts': self.exchange_attempts,
            'total_exchanges_accepted': self.exchanges_accepted,
            'overall_acceptance_rate': self.exchanges_accepted / max(1, self.exchange_attempts),
            'replica_state_probabilities': replica_probs.tolist(),
            'average_round_trip_time': np.mean(round_trip_times) if round_trip_times else 0,
            'round_trip_times': round_trip_times[:10]  # First 10 for brevity
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
            -np.pi,      # minValue
            np.pi,       # maxValue  
            0.35,        # biasWidth (~20 degrees)
            True         # periodic
        )
        
        bias_variables.append(phi_cv)
        logger.info(f"Added phi dihedral {i+1} as bias variable: atoms {phi_atoms}")
    
    return bias_variables


# Example usage function
def run_remd_simulation(pdb_file: str, 
                       output_dir: str = "remd_output",
                       total_steps: int = 1000,  # VERY FAST for testing
                       equilibration_steps: int = 100,  # Default equilibration steps
                       temperatures: List[float] = None,
                       use_metadynamics: bool = True,
                       checkpoint_manager=None) -> str:
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
        Path to demultiplexed trajectory at 300K
    """
    
    # Stage: Replica Initialization
    if checkpoint_manager and not checkpoint_manager.is_step_completed("replica_initialization"):
        checkpoint_manager.mark_step_started("replica_initialization")
        
    # Set up bias variables if requested
    bias_variables = setup_bias_variables(pdb_file) if use_metadynamics else None
    
    # Create and configure REMD
    remd = ReplicaExchange(
        pdb_file=pdb_file,
        temperatures=temperatures,
        output_dir=output_dir,
        exchange_frequency=50  # Very frequent exchanges for testing
    )
    
    # Set up replicas
    remd.setup_replicas(bias_variables=bias_variables)
    
    # Save state
    if checkpoint_manager:
        checkpoint_manager.save_state({
            'remd_config': {
                'pdb_file': pdb_file,
                'temperatures': remd.temperatures,
                'output_dir': output_dir,
                'exchange_frequency': remd.exchange_frequency,
                'bias_variables': bias_variables
            }
        })
        checkpoint_manager.mark_step_completed("replica_initialization", {
            "n_replicas": remd.n_replicas,
            "temperature_range": f"{min(remd.temperatures):.1f}-{max(remd.temperatures):.1f}K"
        })
    elif checkpoint_manager and checkpoint_manager.is_step_completed("replica_initialization"):
        # Load existing state
        state_data = checkpoint_manager.load_state()
        remd_config = state_data.get('remd_config', {})
        
        # Recreate REMD object
        remd = ReplicaExchange(
            pdb_file=pdb_file,
            temperatures=temperatures,
            output_dir=output_dir,
            exchange_frequency=50
        )
        
        # Set up bias variables if they were used
        bias_variables = remd_config.get('bias_variables') if use_metadynamics else None
        
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
            exchange_frequency=50
        )
        remd.setup_replicas(bias_variables=bias_variables)
    
    # Run simulation with checkpoint integration
    remd.run_simulation(total_steps=total_steps, equilibration_steps=equilibration_steps, checkpoint_manager=checkpoint_manager)
    
    # Demultiplex for analysis (separate step - don't fail the entire simulation)
    demux_traj = None
    if checkpoint_manager and not checkpoint_manager.is_step_completed("trajectory_demux"):
        if checkpoint_manager:
            checkpoint_manager.mark_step_started("trajectory_demux")
        
        # Small delay to ensure DCD files are fully written to disk
        import time
        logger.info("Waiting for DCD files to be fully written...")
        time.sleep(2.0)
        
        try:
            demux_traj = remd.demux_trajectories(target_temperature=300.0, equilibration_steps=equilibration_steps)
            if demux_traj:
                logger.info(f"✓ Demultiplexing successful: {demux_traj}")
                if checkpoint_manager:
                    checkpoint_manager.mark_step_completed("trajectory_demux", {"demux_file": demux_traj})
            else:
                logger.warning("⚠ Demultiplexing returned no trajectory")
                if checkpoint_manager:
                    checkpoint_manager.mark_step_failed("trajectory_demux", "No frames found for demultiplexing")
        except Exception as e:
            logger.warning(f"⚠ Demultiplexing failed: {e}")
            if checkpoint_manager:
                checkpoint_manager.mark_step_failed("trajectory_demux", str(e))
        
        # Always log that the simulation itself was successful
        logger.info("🎉 REMD simulation completed successfully!")
        logger.info("Raw trajectory files are available for manual analysis")
    else:
        logger.info("Trajectory demux already completed or checkpoint manager not available")
    
    # Print statistics
    stats = remd.get_exchange_statistics()
    logger.info(f"REMD Statistics: {stats}")
    
    return demux_traj