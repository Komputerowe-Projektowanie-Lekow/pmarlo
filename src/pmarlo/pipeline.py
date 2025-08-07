# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Pipeline orchestration module for PMARLO.

Provides a simple interface to coordinate protein preparation, replica exchange,
simulation, and Markov state model analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .manager.checkpoint_manager import CheckpointManager
from .markov_state_model.markov_state_model import EnhancedMSM as MarkovStateModel
from .markov_state_model.markov_state_model import run_complete_msm_analysis
from .protein.protein import Protein
from .replica_exchange.replica_exchange import ReplicaExchange, run_remd_simulation
from .simulation.simulation import Simulation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main orchestration class for PMARLO.

    This class provides the high-level interface for coordinating all components
    of the protein simulation and MSM analysis workflow.
    """

    def __init__(
        self,
        pdb_file: str,
        output_dir: str = "pmarlo_output",
        temperatures: Optional[List[float]] = None,
        n_replicas: int = 3,
        steps: int = 1000,
        n_states: int = 50,
        use_replica_exchange: bool = True,
        use_metadynamics: bool = True,
        checkpoint_id: Optional[str] = None,
        auto_continue: bool = True,
        enable_checkpoints: bool = True,
    ):
        """
        Initialize the PMARLO pipeline.

        Args:
            pdb_file: Path to the input PDB file
            output_dir: Directory for all output files
            temperatures: List of temperatures for replica exchange (K)
            n_replicas: Number of replicas for REMD
            steps: Number of simulation steps
            n_states: Number of MSM states
            use_replica_exchange: Whether to use replica exchange
            use_metadynamics: Whether to use metadynamics
            checkpoint_id: Optional checkpoint ID for resuming runs
        """
        self.pdb_file = pdb_file
        self.output_dir = Path(output_dir)
        self.steps = steps
        self.n_states = n_states
        self.use_replica_exchange = use_replica_exchange
        self.use_metadynamics = use_metadynamics

        # Set default temperatures if not provided
        if temperatures is None:
            if use_replica_exchange:
                # Create temperature ladder with small gaps for high exchange rates
                self.temperatures = [300.0 + i * 10.0 for i in range(n_replicas)]
            else:
                self.temperatures = [300.0]
        else:
            self.temperatures = temperatures

        # Initialize components
        self.protein: Optional[Protein] = None
        self.replica_exchange: Optional[ReplicaExchange] = None
        self.simulation: Optional[Simulation] = None
        self.markov_state_model: Optional[MarkovStateModel] = None

        # Paths
        self.prepared_pdb: Optional[Path] = None
        self.trajectory_files: List[str] = []

        # Setup checkpoint manager
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.enable_checkpoints = enable_checkpoints

        if enable_checkpoints:
            # Define pipeline steps based on configuration
            pipeline_steps = ["protein_preparation"]

            if use_replica_exchange:
                pipeline_steps.extend(
                    ["replica_setup", "replica_exchange_simulation", "trajectory_demux"]
                )
            else:
                pipeline_steps.append("simulation")

            pipeline_steps.append("msm_analysis")

            # Auto-detect interrupted runs if no ID provided
            if not checkpoint_id and auto_continue:
                auto_detected = CheckpointManager.auto_detect_interrupted_run(
                    str(self.output_dir)
                )
                if auto_detected:
                    self.checkpoint_manager = auto_detected
                    logger.info(
                        f"Auto-continuing interrupted run: {auto_detected.run_id}"
                    )
                else:
                    self.checkpoint_manager = CheckpointManager(
                        output_base_dir=str(self.output_dir),
                        pipeline_steps=pipeline_steps,
                    )
            else:
                self.checkpoint_manager = CheckpointManager(
                    run_id=checkpoint_id,
                    output_base_dir=str(self.output_dir),
                    pipeline_steps=pipeline_steps,
                    auto_continue=auto_continue,
                )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PMARLO Pipeline initialized")
        logger.info(f"  PDB file: {self.pdb_file}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Temperatures: {self.temperatures}")
        logger.info(f"  Replica Exchange: {self.use_replica_exchange}")
        logger.info(f"  Metadynamics: {self.use_metadynamics}")

    def setup_protein(self, ph: float = 7.0) -> Protein:
        """
        Setup and prepare the protein.

        Args:
            ph: pH for protonation state

        Returns:
            Prepared Protein object
        """
        logger.info("Stage 1/4: Protein Preparation")

        self.protein = Protein(self.pdb_file, ph=ph)

        # Save prepared protein
        self.prepared_pdb = self.output_dir / "prepared_protein.pdb"
        self.protein.save(str(self.prepared_pdb))

        properties = self.protein.get_properties()
        logger.info(
            f"Protein prepared: {properties['num_atoms']} atoms, {properties['num_residues']} residues"
        )

        return self.protein

    def setup_replica_exchange(self) -> Optional[ReplicaExchange]:
        """
        Setup replica exchange if enabled.

        Returns:
            ReplicaExchange object if enabled, None otherwise
        """
        if not self.use_replica_exchange:
            return None

        logger.info("Stage 2/4: Replica Exchange Setup")

        remd_output_dir = self.output_dir / "replica_exchange"

        self.replica_exchange = ReplicaExchange(
            pdb_file=str(self.prepared_pdb),
            temperatures=self.temperatures,
            output_dir=str(remd_output_dir),
        )

        # CRITICAL FIX: Initialize replicas before returning
        # This was the root cause of the IndexError - contexts list was empty
        bias_variables = None
        if self.use_metadynamics:
            # Import here to avoid circular imports
            from .replica_exchange.replica_exchange import setup_bias_variables

            bias_variables = setup_bias_variables(str(self.prepared_pdb))

        self.replica_exchange.setup_replicas(bias_variables=bias_variables)

        logger.info(f"Replica exchange setup with {len(self.temperatures)} replicas")
        return self.replica_exchange

    def setup_simulation(self) -> Simulation:
        """
        Setup simulation.

        Returns:
            Simulation object
        """
        logger.info("Stage 3/4: Simulation Setup")

        sim_output_dir = self.output_dir / "simulation"

        self.simulation = Simulation(
            pdb_file=str(self.prepared_pdb),
            temperature=self.temperatures[
                0
            ],  # Use first temperature for single simulation
            steps=self.steps,
            output_dir=str(sim_output_dir),
            use_metadynamics=self.use_metadynamics,
        )

        logger.info(
            f"Simulation setup for {self.steps} steps at {self.temperatures[0]}K"
        )
        return self.simulation

    def setup_markov_state_model(self) -> MarkovStateModel:
        """
        Setup Markov State Model.

        Returns:
            MarkovStateModel object
        """
        logger.info("Stage 4/4: Markov State Model Setup")

        msm_output_dir = self.output_dir / "msm_analysis"

        self.markov_state_model = MarkovStateModel(output_dir=str(msm_output_dir))

        logger.info(f"MSM setup for {self.n_states} states")
        return self.markov_state_model

    def run(self) -> Dict[str, Any]:
        """
        Run the complete PMARLO pipeline.

        Returns:
            Dictionary containing results and output paths
        """
        logger.info("=" * 60)
        logger.info("STARTING PMARLO PIPELINE")
        logger.info("=" * 60)

        results: Dict[str, Any] = {}

        # Setup checkpoint manager run directory
        if self.checkpoint_manager:
            self.checkpoint_manager.setup_run_directory()
            # Save pipeline configuration
            config = {
                "pdb_file": self.pdb_file,
                "temperatures": self.temperatures,
                "steps": self.steps,
                "n_states": self.n_states,
                "use_replica_exchange": self.use_replica_exchange,
                "use_metadynamics": self.use_metadynamics,
            }
            self.checkpoint_manager.save_config(config)

        try:
            # Stage 1: Protein preparation
            if (
                not self.checkpoint_manager
                or not self.checkpoint_manager.is_step_completed("protein_preparation")
            ):
                if self.checkpoint_manager:
                    self.checkpoint_manager.mark_step_started("protein_preparation")

                protein = self.setup_protein()

                if self.checkpoint_manager:
                    self.checkpoint_manager.mark_step_completed(
                        "protein_preparation",
                        {
                            "prepared_pdb": str(self.prepared_pdb),
                            "properties": protein.get_properties(),
                        },
                    )
            else:
                logger.info("Protein preparation already completed, skipping...")
                # Load prepared protein path from checkpoint
                state = self.checkpoint_manager.load_state()
                prepared_pdb_str = state.get(
                    "prepared_pdb", str(self.output_dir / "prepared_protein.pdb")
                )
                self.prepared_pdb = Path(prepared_pdb_str)
                protein = Protein(str(self.prepared_pdb))

            results["protein"] = {
                "prepared_pdb": str(self.prepared_pdb),
                "properties": protein.get_properties(),
            }

            # Stage 2: Run simulations
            if self.use_replica_exchange:
                step_name = "replica_exchange_simulation"
                if (
                    not self.checkpoint_manager
                    or not self.checkpoint_manager.is_step_completed(step_name)
                ):
                    if self.checkpoint_manager:
                        self.checkpoint_manager.mark_step_started(step_name)

                    # Setup and run replica exchange
                    remd = self.setup_replica_exchange()
                    if remd is not None:
                        trajectory_files = remd.run_simulation(self.steps)
                    else:
                        trajectory_files = {}

                    if self.checkpoint_manager:
                        self.checkpoint_manager.save_state(
                            {
                                "prepared_pdb": str(self.prepared_pdb),
                                "trajectory_files": trajectory_files,
                            }
                        )
                        self.checkpoint_manager.mark_step_completed(
                            step_name, {"trajectory_files": trajectory_files}
                        )
                else:
                    logger.info(
                        "Replica exchange simulation already completed, loading results..."
                    )
                    state = self.checkpoint_manager.load_state()
                    trajectory_files = state.get("trajectory_files", {})

                # Use demultiplexed trajectory for MSM analysis
                if isinstance(trajectory_files, dict) and "demuxed" in trajectory_files:
                    self.trajectory_files = [trajectory_files["demuxed"]]
                    analysis_temperatures = [self.temperatures[0]]
                else:
                    # Use all replica trajectories
                    self.trajectory_files = (
                        list(trajectory_files.values())
                        if isinstance(trajectory_files, dict)
                        else trajectory_files
                    )
                    analysis_temperatures = self.temperatures

                results["replica_exchange"] = {
                    "trajectory_files": self.trajectory_files,
                    "temperatures": [
                        str(t) for t in self.temperatures
                    ],  # Convert to strings
                    "output_dir": str(self.output_dir / "replica_exchange"),
                }

            else:
                step_name = "simulation"
                if (
                    not self.checkpoint_manager
                    or not self.checkpoint_manager.is_step_completed(step_name)
                ):
                    if self.checkpoint_manager:
                        self.checkpoint_manager.mark_step_started(step_name)

                    # Setup and run single simulation
                    sim = self.setup_simulation()
                    trajectory_file, states = sim.run_complete_simulation()
                    self.trajectory_files = [trajectory_file]
                    analysis_temperatures = [self.temperatures[0]]

                    if self.checkpoint_manager:
                        self.checkpoint_manager.save_state(
                            {
                                "prepared_pdb": str(self.prepared_pdb),
                                "trajectory_file": trajectory_file,
                                "states": states,
                            }
                        )
                        self.checkpoint_manager.mark_step_completed(
                            step_name,
                            {"trajectory_file": trajectory_file, "states": states},
                        )
                else:
                    logger.info("Simulation already completed, loading results...")
                    state = self.checkpoint_manager.load_state()
                    trajectory_file_from_state = state.get("trajectory_file")
                    states_from_state = state.get("states")
                    if trajectory_file_from_state is not None:
                        trajectory_file = str(trajectory_file_from_state)
                    else:
                        trajectory_file = ""
                    if states_from_state is not None:
                        states = np.array(states_from_state)
                    else:
                        states = np.array([])
                    self.trajectory_files = [trajectory_file]
                    analysis_temperatures = [self.temperatures[0]]

                results["simulation"] = {
                    "trajectory_file": trajectory_file,
                    "states": states.tolist() if len(states) > 0 else [],
                    "output_dir": str(self.output_dir / "simulation"),
                }

            # Stage 3: MSM Analysis
            step_name = "msm_analysis"
            if (
                not self.checkpoint_manager
                or not self.checkpoint_manager.is_step_completed(step_name)
            ):
                if self.checkpoint_manager:
                    self.checkpoint_manager.mark_step_started(step_name)

                msm = self.setup_markov_state_model()

                # Run MSM analysis with collected trajectories
                if hasattr(msm, "run_complete_analysis"):
                    msm_results = msm.run_complete_analysis(
                        trajectory_files=self.trajectory_files,
                        topology_file=str(self.prepared_pdb),
                        n_clusters=self.n_states,
                        temperatures=analysis_temperatures,
                    )
                else:
                    # Fallback for basic analysis
                    logger.warning("Using basic MSM analysis")
                    msm_results = {"warning": "Basic analysis only"}

                if self.checkpoint_manager:
                    self.checkpoint_manager.mark_step_completed(
                        step_name,
                        {
                            "n_clusters": str(self.n_states),
                            "analysis_results": msm_results,
                        },
                    )
            else:
                logger.info("MSM analysis already completed, loading results...")
                checkpoint_data = None
                for step in self.checkpoint_manager.life_data["completed_steps"]:
                    if step["name"] == step_name:
                        checkpoint_data = step["metadata"]
                        break
                msm_results = (
                    checkpoint_data.get("analysis_results", {})
                    if checkpoint_data
                    else {}
                )

            results["msm"] = {
                "output_dir": str(self.output_dir / "msm_analysis"),
                "n_states": str(self.n_states),  # Convert to string
                "results": msm_results,
            }

            # Final summary
            results["pipeline"] = {
                "status": "completed",
                "output_dir": str(self.output_dir),
                "use_replica_exchange": str(
                    self.use_replica_exchange
                ),  # Convert to string
                "use_metadynamics": str(self.use_metadynamics),  # Convert to string
                "steps": str(self.steps),  # Convert to string
                "temperatures": [
                    str(t) for t in self.temperatures
                ],  # Convert to strings
            }

            # Mark pipeline as completed
            if self.checkpoint_manager:
                self.checkpoint_manager.life_data["status"] = "completed"
                self.checkpoint_manager.save_life_data()

            logger.info("=" * 60)
            logger.info("PMARLO PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback

            traceback.print_exc()

            # Mark current step as failed
            if self.checkpoint_manager:
                current_stage = self.checkpoint_manager.life_data.get(
                    "current_stage", "unknown"
                )
                self.checkpoint_manager.mark_step_failed(current_stage, str(e))

            results["pipeline"] = {
                "status": "failed",
                "error": str(e),
                "output_dir": str(self.output_dir),
                "checkpoint_id": (
                    self.checkpoint_manager.run_id
                    if self.checkpoint_manager
                    else str(None)
                ),
            }

            return results

    def get_components(self) -> Dict[str, Any]:
        """
        Get all initialized components.

        Returns:
            Dictionary of initialized components
        """
        return {
            "protein": self.protein,
            "replica_exchange": self.replica_exchange,
            "simulation": self.simulation,
            "markov_state_model": self.markov_state_model,
            "checkpoint_manager": self.checkpoint_manager,
        }

    def get_checkpoint_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current checkpoint status.

        Returns:
            Checkpoint status dictionary or None if checkpointing disabled
        """
        if self.checkpoint_manager:
            return self.checkpoint_manager.print_status(verbose=False)
        return None

    def can_continue(self) -> bool:
        """
        Check if this pipeline can be continued from a checkpoint.

        Returns:
            True if pipeline can be continued, False otherwise
        """
        if self.checkpoint_manager:
            return bool(self.checkpoint_manager.can_continue())
        return False


# Convenience function for the 5-line API
def run_pmarlo(
    pdb_file: str,
    temperatures: Optional[List[float]] = None,
    steps: int = 1000,
    n_states: int = 50,
    output_dir: str = "pmarlo_output",
    checkpoint_id: Optional[str] = None,
    auto_continue: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run complete PMARLO pipeline in one function call.

    This is the main convenience function for the 5-line API.

    Args:
        pdb_file: Path to input PDB file
        temperatures: List of temperatures for replica exchange
        steps: Number of simulation steps
        n_states: Number of MSM states
        output_dir: Output directory
        **kwargs: Additional arguments for Pipeline

    Returns:
        Dictionary containing all results
    """
    pipeline = Pipeline(
        pdb_file=pdb_file,
        temperatures=temperatures,
        steps=steps,
        n_states=n_states,
        output_dir=output_dir,
        checkpoint_id=checkpoint_id,
        auto_continue=auto_continue,
        **kwargs,
    )

    return pipeline.run()


class LegacyPipeline:
    """
    Legacy pipeline implementation with checkpoint support.

    This maintains compatibility with the original REMD + Enhanced MSM pipeline
    while providing checkpoint and resume functionality.
    """

    def __init__(
        self,
        pdb_file: str,
        output_dir: str = "output",
        run_id: Optional[str] = None,
        continue_run: bool = False,
    ):
        """
        Initialize the legacy pipeline.

        Args:
            pdb_file: Path to input PDB file
            output_dir: Base output directory
            run_id: Optional run ID for checkpointing
            continue_run: Whether to continue from existing run
        """
        self.pdb_file = pdb_file
        self.output_base_dir = Path(output_dir)
        self.run_id = run_id
        self.continue_run = continue_run
        self.checkpoint_manager: Optional[CheckpointManager] = None

    def run_legacy_remd_pipeline(
        self, steps: int = 1000, n_states: int = 50
    ) -> Optional[Path]:
        """Run the legacy REMD + Enhanced MSM pipeline with checkpoint support."""

        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        if self.continue_run and self.run_id:
            # Load existing run
            try:
                self.checkpoint_manager = CheckpointManager.load_existing_run(
                    self.run_id, str(self.output_base_dir)
                )
                print(f"Resuming run {self.run_id}...")
                self.checkpoint_manager.print_status()
            except FileNotFoundError:
                print(f"Error: No existing run found with ID {self.run_id}")
                print(f"Looking in: {self.output_base_dir}")
                from .manager.checkpoint_manager import list_runs

                list_runs(str(self.output_base_dir))
                return None
        elif self.continue_run and not self.run_id:
            # List available runs and ask user to specify
            print("Error: --continue requires --id to specify which run to continue")
            from .manager.checkpoint_manager import list_runs

            list_runs(str(self.output_base_dir))
            return None
        else:
            # Start new run
            self.checkpoint_manager = CheckpointManager(
                self.run_id, str(self.output_base_dir)
            )
            self.checkpoint_manager.setup_run_directory()
            print(f"Started new run with ID: {self.checkpoint_manager.run_id}")

        # File paths
        pdb_file = Path(self.pdb_file)
        pdb_fixed_path = self.checkpoint_manager.run_dir / "inputs" / "prepared.pdb"
        remd_output_dir = self.checkpoint_manager.run_dir / "trajectories"
        msm_output_dir = self.checkpoint_manager.run_dir / "analysis"

        # Save configuration
        config = {
            "pdb_file": str(pdb_file),
            "steps": steps,
            "n_states": n_states,
            "temperatures": [
                300.0,
                310.0,
                320.0,
            ],  # 3 replicas with small 10K gaps for high exchange rates
            "use_metadynamics": True,
            "created_at": self.checkpoint_manager.life_data["created"],
        }
        self.checkpoint_manager.save_config(config)

        try:
            print("=" * 60)
            print("LEGACY REPLICA EXCHANGE + ENHANCED MSM PIPELINE")
            print("=" * 60)

            # Use checkpoint manager to determine what to run next
            while True:
                next_step = self.checkpoint_manager.get_next_step()

                if next_step is None:
                    print("\n🎉 All steps completed!")
                    break

                # Clear failed status when retrying a step
                if next_step in [
                    s.get("name")
                    for s in self.checkpoint_manager.life_data["failed_steps"]
                ]:
                    print(f"\n🔄 Retrying failed step: {next_step}")
                    self.checkpoint_manager.clear_failed_step(next_step)

                # Execute the appropriate step
                if next_step == "protein_preparation":
                    self.checkpoint_manager.mark_step_started("protein_preparation")
                    print("\n[Stage 1/6] Protein Preparation...")

                    protein = Protein(str(pdb_file), ph=7.0)

                    # Ensure the inputs directory exists before saving
                    pdb_fixed_path.parent.mkdir(parents=True, exist_ok=True)

                    protein.save(str(pdb_fixed_path))
                    properties = protein.get_properties()
                    print(
                        f"Protein prepared: {properties['num_atoms']} atoms, {properties['num_residues']} residues"
                    )

                    # Copy input files for reproducibility
                    self.checkpoint_manager.copy_input_files([str(pdb_file)])

                    self.checkpoint_manager.mark_step_completed(
                        "protein_preparation",
                        {
                            "num_atoms": properties["num_atoms"],
                            "num_residues": properties["num_residues"],
                            "pdb_fixed_path": str(pdb_fixed_path),
                        },
                    )

                elif next_step == "system_setup":
                    self.checkpoint_manager.mark_step_started("system_setup")
                    print("\n[Stage 2/6] System Setup...")
                    print("Setting up temperature ladder for enhanced sampling...")

                    # Just mark as completed - the actual setup happens in replica_initialization
                    self.checkpoint_manager.mark_step_completed(
                        "system_setup",
                        {
                            "temperatures": config["temperatures"],
                            "use_metadynamics": config["use_metadynamics"],
                        },
                    )

                elif next_step in [
                    "replica_initialization",
                    "energy_minimization",
                    "gradual_heating",
                    "equilibration",
                    "production_simulation",
                    "trajectory_demux",
                ]:
                    print("\n[Stage 3/6] Replica Exchange Molecular Dynamics...")

                    # Run REMD simulation with checkpoint integration
                    demux_trajectory = run_remd_simulation(
                        pdb_file=str(pdb_fixed_path),
                        output_dir=str(remd_output_dir),
                        total_steps=steps,
                        temperatures=config["temperatures"],
                        use_metadynamics=config["use_metadynamics"],
                        checkpoint_manager=self.checkpoint_manager,  # Pass checkpoint manager
                    )

                    print(
                        f"REMD completed. Demultiplexed trajectory: {demux_trajectory}"
                    )
                    # The REMD function handles its own checkpoints, so we continue the loop

                elif next_step == "trajectory_analysis":
                    self.checkpoint_manager.mark_step_started("trajectory_analysis")
                    print("\n[Stage 4/6] Enhanced Markov State Model Analysis...")

                    # Reconstruct demux trajectory path
                    demux_trajectory = str(remd_output_dir / "demuxed_trajectory.dcd")

                    # Use demultiplexed trajectory if available, otherwise use all trajectories
                    if demux_trajectory and Path(demux_trajectory).exists():
                        trajectory_files = [demux_trajectory]
                        analysis_temperatures = [300.0]  # Only target temperature
                    else:
                        # Use all replica trajectories for TRAM analysis
                        trajectory_files = [
                            str(remd_output_dir / f"replica_{i:02d}.dcd")
                            for i in range(len(config["temperatures"]))
                        ]
                        trajectory_files = [
                            f for f in trajectory_files if Path(f).exists()
                        ]
                        analysis_temperatures = config["temperatures"]

                    if not trajectory_files:
                        raise ValueError("No trajectory files found for analysis")

                    print(f"Analyzing {len(trajectory_files)} trajectories...")

                    # Run complete MSM analysis
                    msm = run_complete_msm_analysis(
                        trajectory_files=trajectory_files,
                        topology_file=str(pdb_fixed_path),
                        output_dir=str(msm_output_dir),
                        n_clusters=n_states,
                        lag_time=10,
                        feature_type="phi_psi",
                        temperatures=analysis_temperatures,
                    )

                    self.checkpoint_manager.mark_step_completed(
                        "trajectory_analysis",
                        {
                            "n_trajectories": len(trajectory_files),
                            "n_clusters": n_states,
                            "analysis_output": str(msm_output_dir),
                        },
                    )

                else:
                    print(f"Unknown step: {next_step}")
                    break

            # Final summary
            print("\n[Stage 5/6] Pipeline Complete!")
            print(f"✓ Results saved to: {self.checkpoint_manager.run_dir}")
            print("✓ Ready for analysis and visualization")

            # Mark pipeline as completed
            self.checkpoint_manager.life_data["status"] = "completed"
            self.checkpoint_manager.save_life_data()

            # Print final status
            self.checkpoint_manager.print_status()

            return self.checkpoint_manager.run_dir

        except Exception as e:
            if self.checkpoint_manager:
                self.checkpoint_manager.mark_step_failed(
                    self.checkpoint_manager.life_data["current_stage"], str(e)
                )
            print(f"An error occurred in REMD pipeline: {e}")
            import traceback

            traceback.print_exc()
            print("\nCheckpoint saved. You can resume with:")
            if self.checkpoint_manager:
                print(
                    f"python main.py --mode remd --id {self.checkpoint_manager.run_id} --continue"
                )
            return None
