# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Pipeline orchestration module for PMARLO.

Provides a simple interface to coordinate protein preparation, replica exchange,
simulation, and Markov state model analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .manager.checkpoint_manager import CheckpointManager
from .markov_state_model.enhanced_msm import EnhancedMSM as MarkovStateModel
from .markov_state_model.enhanced_msm import run_complete_msm_analysis
from .protein.protein import Protein
from .replica_exchange.config import RemdConfig
from .replica_exchange.replica_exchange import ReplicaExchange, run_remd_simulation
from .simulation.simulation import Simulation
from .utils.seed import set_global_seed

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
        output_dir: str = "output",
        temperatures: Optional[List[float]] = None,
        n_replicas: int = 3,
        steps: int = 1000,
        n_states: int = 50,
        use_replica_exchange: bool = True,
        use_metadynamics: bool = True,
        checkpoint_id: Optional[str] = None,
        auto_continue: bool = True,
        enable_checkpoints: bool = True,
        random_state: int | None = None,
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
            random_state: Seed for reproducible behaviour across components.
        """
        self.pdb_file = pdb_file
        self.output_dir = Path(output_dir)
        self.steps = steps
        self.n_states = n_states
        self.use_replica_exchange = use_replica_exchange
        self.use_metadynamics = use_metadynamics
        self.random_state = random_state

        if random_state is not None:
            set_global_seed(int(random_state))

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

        logger.info("PMARLO Pipeline initialized")
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
            "Protein prepared: "
            f"{properties['num_atoms']} atoms, "
            f"{properties['num_residues']} residues"
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

        self.replica_exchange = ReplicaExchange.from_config(
            RemdConfig(
                pdb_file=str(self.prepared_pdb),
                temperatures=self.temperatures,
                output_dir=str(remd_output_dir),
            )
        )

        # CRITICAL FIX: Initialize replicas before returning
        # This was the root cause of the IndexError - contexts list was empty
        bias_variables = None
        if self.use_metadynamics:
            # Import here to avoid circular imports
            from .replica_exchange.replica_exchange import setup_bias_variables

            bias_variables = setup_bias_variables(str(self.prepared_pdb))

        # Plan stride for pipeline run before reporters are created
        self.replica_exchange.plan_reporter_stride(
            total_steps=int(self.steps),
            equilibration_steps=min(self.steps // 10, 200),
            target_frames=5000,
        )
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
            # Ensure DCD is produced even for short test runs
            dcd_stride=max(1, min(100, self.steps // 10) if self.steps else 10),
        )

        logger.info(
            "Simulation setup for " f"{self.steps} steps at {self.temperatures[0]}K"
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

        self._ensure_checkpoint_dir()
        self._save_pipeline_config()

        try:
            protein = self._stage_protein()
            results["protein"] = {
                "prepared_pdb": str(self.prepared_pdb),
                "properties": protein.get_properties(),
            }

            if self.use_replica_exchange:
                trajectory_files = self._stage_replica_exchange_simulation()
                analysis_temperatures = self._determine_analysis_temperatures(
                    trajectory_files
                )
                results["replica_exchange"] = {
                    "trajectory_files": self.trajectory_files,
                    "temperatures": [str(t) for t in self.temperatures],
                    "output_dir": str(self.output_dir / "replica_exchange"),
                }
            else:
                trajectory_file, states = self._stage_single_simulation()
                analysis_temperatures = [self.temperatures[0]]
                results["simulation"] = {
                    "trajectory_file": trajectory_file,
                    "states": states.tolist() if len(states) > 0 else [],
                    "output_dir": str(self.output_dir / "simulation"),
                }

            msm_results = self._stage_msm_analysis(analysis_temperatures)
            results["msm"] = {
                "output_dir": str(self.output_dir / "msm_analysis"),
                "n_states": str(self.n_states),
                "results": msm_results,
            }

            self._finalize_success(results)
            return results
        except Exception as e:  # noqa: BLE001 - broad to record and checkpoint
            return self._handle_run_exception(e, results)

    # ---- Helper methods to reduce complexity of run() ----

    def _ensure_checkpoint_dir(self) -> None:
        if self.checkpoint_manager:
            self.checkpoint_manager.setup_run_directory()

    def _save_pipeline_config(self) -> None:
        if not self.checkpoint_manager:
            return
        config = {
            "pdb_file": self.pdb_file,
            "temperatures": self.temperatures,
            "steps": self.steps,
            "n_states": self.n_states,
            "use_replica_exchange": self.use_replica_exchange,
            "use_metadynamics": self.use_metadynamics,
        }
        self.checkpoint_manager.save_config(config)

    def _stage_protein(self) -> Protein:
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
            return protein
        logger.info("Protein preparation already completed, skipping...")
        state = self.checkpoint_manager.load_state() if self.checkpoint_manager else {}
        prepared_pdb_str = state.get(
            "prepared_pdb", str(self.output_dir / "prepared_protein.pdb")
        )
        self.prepared_pdb = Path(prepared_pdb_str)
        return Protein(str(self.prepared_pdb))

    def _stage_replica_exchange_simulation(self) -> Dict[str, Any] | List[str]:
        step_name = "replica_exchange_simulation"
        if (
            not self.checkpoint_manager
            or not self.checkpoint_manager.is_step_completed(step_name)
        ):
            if self.checkpoint_manager:
                self.checkpoint_manager.mark_step_started(step_name)
            remd = self.setup_replica_exchange()
            trajectory_files: Dict[str, Any] | List[str]
            if remd is not None:
                # demux path or None is returned; downstream handles both dict/list
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
                "Replica exchange simulation " "already completed, loading results..."
            )
            state = (
                self.checkpoint_manager.load_state() if self.checkpoint_manager else {}
            )
            from typing import cast

            trajectory_files = cast(
                Dict[str, Any] | List[str], state.get("trajectory_files", {})
            )
        return trajectory_files

    def _determine_analysis_temperatures(
        self, trajectory_files: Dict[str, Any] | List[str]
    ) -> List[float]:
        if isinstance(trajectory_files, dict) and "demuxed" in trajectory_files:
            self.trajectory_files = [trajectory_files["demuxed"]]
            return [self.temperatures[0]]
        self.trajectory_files = (
            list(trajectory_files.values())
            if isinstance(trajectory_files, dict)
            else trajectory_files
        )
        return self.temperatures

    def _stage_single_simulation(self) -> tuple[str, np.ndarray]:
        step_name = "simulation"
        if (
            not self.checkpoint_manager
            or not self.checkpoint_manager.is_step_completed(step_name)
        ):
            if self.checkpoint_manager:
                self.checkpoint_manager.mark_step_started(step_name)
            sim = self.setup_simulation()
            trajectory_file, states = sim.run_complete_simulation()
            self.trajectory_files = [trajectory_file]
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
            return trajectory_file, states
        logger.info("Simulation already completed, loading results...")
        state = self.checkpoint_manager.load_state() if self.checkpoint_manager else {}
        trajectory_file_from_state = state.get("trajectory_file")
        states_from_state = state.get("states")
        trajectory_file_loaded: str = (
            str(trajectory_file_from_state)
            if trajectory_file_from_state is not None
            else ""
        )
        states_loaded = (
            np.array(states_from_state)
            if states_from_state is not None
            else np.array([])
        )
        self.trajectory_files = [trajectory_file_loaded]
        return trajectory_file_loaded, states_loaded

    def _stage_msm_analysis(self, analysis_temperatures: List[float]) -> Dict[str, Any]:
        step_name = "msm_analysis"
        if (
            not self.checkpoint_manager
            or not self.checkpoint_manager.is_step_completed(step_name)
        ):
            if self.checkpoint_manager:
                self.checkpoint_manager.mark_step_started(step_name)
            msm = self.setup_markov_state_model()
            # Ensure prepared PDB exists before analysis
            if self.prepared_pdb is None:
                raise RuntimeError("Prepared PDB not set before MSM analysis")
            msm_results: Dict[str, Any]
            if hasattr(msm, "run_complete_analysis"):
                from typing import cast

                msm_results = cast(
                    Dict[str, Any],
                    msm.run_complete_analysis(
                        trajectory_files=self.trajectory_files,
                        topology_file=str(self.prepared_pdb),
                        n_clusters=self.n_states,
                        temperatures=analysis_temperatures,
                    ),
                )
            else:
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
            return msm_results
        logger.info("MSM analysis already completed, loading results...")
        checkpoint_data: Optional[Dict[str, Any]] = None
        if not self.checkpoint_manager:
            return {}
        for step in self.checkpoint_manager.life_data.get("completed_steps", []):
            if step.get("name") == step_name:
                checkpoint_data = step.get("metadata")
                break
        return checkpoint_data.get("analysis_results", {}) if checkpoint_data else {}

    def _finalize_success(self, results: Dict[str, Any]) -> None:
        results["pipeline"] = {
            "status": "completed",
            "output_dir": str(self.output_dir),
            "use_replica_exchange": str(self.use_replica_exchange),
            "use_metadynamics": str(self.use_metadynamics),
            "steps": str(self.steps),
            "temperatures": [str(t) for t in self.temperatures],
        }
        if self.checkpoint_manager:
            cm = self.checkpoint_manager
            cm.life_data["status"] = "completed"
            cm.save_life_data()
        logger.info("=" * 60)
        logger.info("PMARLO PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("=" * 60)

    def _handle_run_exception(
        self, e: Exception, results: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
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
                self.checkpoint_manager.run_id if self.checkpoint_manager else str(None)
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
    output_dir: str = "output",
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

        self._legacy_setup_logging()
        cm = self._legacy_load_or_start_run()
        if cm is None:
            return None
        self.checkpoint_manager = cm

        (
            pdb_file,
            pdb_fixed_path,
            remd_output_dir,
            msm_output_dir,
        ) = self._legacy_init_paths()

        config = self._legacy_save_config(steps, n_states, pdb_file)

        try:
            print("=" * 60)
            print("LEGACY REPLICA EXCHANGE + ENHANCED MSM PIPELINE")
            print("=" * 60)

            while True:
                next_step = self.checkpoint_manager.get_next_step()
                if next_step is None:
                    print("\n🎉 All steps completed!")
                    break
                if next_step in [
                    s.get("name")
                    for s in self.checkpoint_manager.life_data["failed_steps"]
                ]:
                    print(f"\n🔄 Retrying failed step: {next_step}")
                    self.checkpoint_manager.clear_failed_step(next_step)

                self._legacy_dispatch_step(
                    next_step,
                    pdb_file,
                    pdb_fixed_path,
                    remd_output_dir,
                    msm_output_dir,
                    steps,
                    n_states,
                    config,
                )

            self._legacy_finalize_success()
            return self.checkpoint_manager.run_dir
        except Exception as e:  # noqa: BLE001
            self._legacy_handle_exception(e)
            return None

    # ---- Legacy helpers to reduce complexity ----

    def _legacy_setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def _legacy_load_or_start_run(self) -> Optional[CheckpointManager]:
        if self.continue_run and self.run_id:
            try:
                cm = CheckpointManager.load_existing_run(
                    self.run_id, str(self.output_base_dir)
                )
                print(f"Resuming run {self.run_id}...")
                cm.print_status()
                return cm
            except FileNotFoundError:
                print(f"Error: No existing run found with ID {self.run_id}")
                print(f"Looking in: {self.output_base_dir}")
                from .manager.checkpoint_manager import list_runs

                list_runs(str(self.output_base_dir))
                return None
        if self.continue_run and not self.run_id:
            print("Error: --continue requires --id to specify which run to continue")
            from .manager.checkpoint_manager import list_runs

            list_runs(str(self.output_base_dir))
            return None
        cm = CheckpointManager(self.run_id, str(self.output_base_dir))
        cm.setup_run_directory()
        print(f"Started new run with ID: {cm.run_id}")
        return cm

    def _legacy_init_paths(self) -> tuple[Path, Path, Path, Path]:
        pdb_file = Path(self.pdb_file)
        cm = self.checkpoint_manager
        if cm is None:
            # Fallback to default directories under base output directory when no CM
            pdb_fixed_path = self.output_base_dir / "inputs" / "prepared.pdb"
            remd_output_dir = self.output_base_dir / "trajectories"
            msm_output_dir = self.output_base_dir / "analysis"
            return pdb_file, pdb_fixed_path, remd_output_dir, msm_output_dir
        pdb_fixed_path = cm.run_dir / "inputs" / "prepared.pdb"
        remd_output_dir = cm.run_dir / "trajectories"
        msm_output_dir = cm.run_dir / "analysis"
        return pdb_file, pdb_fixed_path, remd_output_dir, msm_output_dir

    def _legacy_save_config(
        self, steps: int, n_states: int, pdb_file: Path
    ) -> Dict[str, Any]:
        cm = self.checkpoint_manager
        base_config: Dict[str, Any] = {
            "pdb_file": str(pdb_file),
            "steps": steps,
            "n_states": n_states,
            "temperatures": [300.0, 310.0, 320.0],
            # 3 replicas with small 10K gaps for high exchange rates
            "use_metadynamics": True,
        }
        if cm is None:
            return {**base_config, "created_at": ""}
        config = {**base_config, "created_at": cm.life_data["created"]}
        cm.save_config(config)
        return config

    def _legacy_dispatch_step(
        self,
        next_step: str,
        pdb_file: Path,
        pdb_fixed_path: Path,
        remd_output_dir: Path,
        msm_output_dir: Path,
        steps: int,
        n_states: int,
        config: Dict,
    ) -> None:
        if next_step == "protein_preparation":
            self._legacy_step_protein_preparation(pdb_file, pdb_fixed_path)
        elif next_step == "system_setup":
            self._legacy_step_system_setup(config)
        elif next_step in [
            "replica_initialization",
            "energy_minimization",
            "gradual_heating",
            "equilibration",
            "production_simulation",
            "trajectory_demux",
        ]:
            demux_trajectory = self._legacy_run_remd(
                pdb_fixed_path, remd_output_dir, steps, config
            )
            print("REMD completed. Demultiplexed trajectory: " f"{demux_trajectory}")
        elif next_step == "trajectory_analysis":
            self._legacy_step_trajectory_analysis(
                pdb_fixed_path, remd_output_dir, msm_output_dir, n_states, config
            )
        else:
            print(f"Unknown step: {next_step}")

    def _legacy_step_protein_preparation(
        self, pdb_file: Path, pdb_fixed_path: Path
    ) -> None:
        cm = self.checkpoint_manager
        if cm is not None:
            cm.mark_step_started("protein_preparation")
        print("\n[Stage 1/6] Protein Preparation...")
        protein = Protein(str(pdb_file), ph=7.0)
        pdb_fixed_path.parent.mkdir(parents=True, exist_ok=True)
        protein.save(str(pdb_fixed_path))
        properties = protein.get_properties()
        print(
            "Protein prepared: "
            f"{properties['num_atoms']} atoms, "
            f"{properties['num_residues']} residues"
        )
        if cm is not None:
            cm.copy_input_files([str(pdb_file)])
            cm.mark_step_completed(
                "protein_preparation",
                {
                    "num_atoms": properties["num_atoms"],
                    "num_residues": properties["num_residues"],
                    "pdb_fixed_path": str(pdb_fixed_path),
                },
            )

    def _legacy_step_system_setup(self, config: Dict) -> None:
        cm = self.checkpoint_manager
        if cm is not None:
            cm.mark_step_started("system_setup")
        print("\n[Stage 2/6] System Setup...")
        print("Setting up temperature ladder for enhanced sampling...")
        # Just mark as completed - actual setup happens in replica_initialization
        if cm is not None:
            cm.mark_step_completed(
                "system_setup",
                {
                    "temperatures": config["temperatures"],
                    "use_metadynamics": config["use_metadynamics"],
                },
            )

    def _legacy_run_remd(
        self, pdb_fixed_path: Path, remd_output_dir: Path, steps: int, config: Dict
    ) -> Optional[str]:
        demux_trajectory = run_remd_simulation(
            pdb_file=str(pdb_fixed_path),
            output_dir=str(remd_output_dir),
            total_steps=steps,
            temperatures=config["temperatures"],
            use_metadynamics=config["use_metadynamics"],
            checkpoint_manager=self.checkpoint_manager,
        )
        return demux_trajectory

    def _legacy_step_trajectory_analysis(
        self,
        pdb_fixed_path: Path,
        remd_output_dir: Path,
        msm_output_dir: Path,
        n_states: int,
        config: Dict,
    ) -> None:
        cm = self.checkpoint_manager
        if cm is not None:
            cm.mark_step_started("trajectory_analysis")
        print("\n[Stage 4/6] Enhanced Markov State Model Analysis...")
        demux_trajectory = str(remd_output_dir / "demuxed_trajectory.dcd")
        if demux_trajectory and Path(demux_trajectory).exists():
            trajectory_files = [demux_trajectory]
            analysis_temperatures = [300.0]
        else:
            trajectory_files = [
                str(remd_output_dir / f"replica_{i:02d}.dcd")
                for i in range(len(config["temperatures"]))
            ]
            trajectory_files = [f for f in trajectory_files if Path(f).exists()]
            analysis_temperatures = config["temperatures"]
        if not trajectory_files:
            raise ValueError("No trajectory files found for analysis")
        print(f"Analyzing {len(trajectory_files)} trajectories...")
        run_complete_msm_analysis(
            trajectory_files=trajectory_files,
            topology_file=str(pdb_fixed_path),
            output_dir=str(msm_output_dir),
            n_states=n_states,
            lag_time=10,
            feature_type="phi_psi",
            temperatures=analysis_temperatures,
        )
        if cm is not None:
            cm.mark_step_completed(
                "trajectory_analysis",
                {
                    "n_trajectories": len(trajectory_files),
                    "n_clusters": n_states,
                    "analysis_output": str(msm_output_dir),
                },
            )

    def _legacy_finalize_success(self) -> None:
        print("\n[Stage 5/6] Pipeline Complete!")
        cm = self.checkpoint_manager
        if cm is not None:
            print(f"✓ Results saved to: {cm.run_dir}")
            print("✓ Ready for analysis and visualization")
            cm.life_data["status"] = "completed"
            cm.save_life_data()
            cm.print_status()
        else:
            print("(No checkpoint manager available)")

    def _legacy_handle_exception(self, e: Exception) -> None:
        cm = self.checkpoint_manager
        if cm is not None:
            cm.mark_step_failed(cm.life_data["current_stage"], str(e))
        print(f"An error occurred in REMD pipeline: {e}")
        import traceback

        traceback.print_exc()
        print("\nCheckpoint saved. You can resume with:")
        if self.checkpoint_manager:
            print(
                "python main.py --mode remd --id "
                f"{self.checkpoint_manager.run_id} --continue"
            )
