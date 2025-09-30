# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Pipeline orchestration module for PMARLO.

Provides a simple interface to coordinate protein preparation, replica exchange,
simulation, and Markov state model analysis using the transform runner system.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..markov_state_model.enhanced_msm import EnhancedMSM as MarkovStateModel
from ..markov_state_model.enhanced_msm import (
    EnhancedMSMProtocol,
)
from ..protein.protein import Protein
from ..replica_exchange.config import RemdConfig
from ..replica_exchange.replica_exchange import ReplicaExchange
from ..replica_exchange.simulation import Simulation
from ..utils.seed import set_global_seed
from .plan import TransformPlan, TransformStep
from .runner import apply_plan

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main orchestration class for PMARLO using transform runner system.

    This class provides the high-level interface for coordinating all components
    of the protein simulation and MSM analysis workflow with built-in checkpointing.
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
            auto_continue: Whether to automatically continue interrupted runs
            enable_checkpoints: Whether to enable checkpointing
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
        self.markov_state_model: Optional[EnhancedMSMProtocol] = None

        # Paths
        self.prepared_pdb: Optional[Path] = None
        self.trajectory_files: List[str] = []

        # Setup transform-based checkpointing
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_id = checkpoint_id
        self.auto_continue = auto_continue

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("PMARLO Pipeline initialized")
        logger.info(f"  PDB file: {self.pdb_file}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Temperatures: {self.temperatures}")
        logger.info(f"  Replica Exchange: {self.use_replica_exchange}")
        logger.info(f"  Metadynamics: {self.use_metadynamics}")
        logger.info(f"  Checkpoints enabled: {self.enable_checkpoints}")

    def _build_transform_plan(self) -> TransformPlan:
        """Build a transform plan based on pipeline configuration."""
        steps = []

        # Protein preparation
        steps.append(
            TransformStep(
                name="PROTEIN_PREPARATION", params={"pdb_file": self.pdb_file}
            )
        )

        if self.use_replica_exchange:
            # Replica exchange pipeline
            steps.extend(
                [
                    TransformStep(name="SYSTEM_SETUP", params={}),
                    TransformStep(
                        name="REPLICA_INITIALIZATION",
                        params={
                            "temperatures": self.temperatures,
                            "output_dir": str(self.output_dir),
                        },
                    ),
                    TransformStep(name="ENERGY_MINIMIZATION", params={}),
                    TransformStep(name="GRADUAL_HEATING", params={}),
                    TransformStep(name="EQUILIBRATION", params={}),
                    TransformStep(
                        name="PRODUCTION_SIMULATION", params={"steps": self.steps}
                    ),
                    TransformStep(name="TRAJECTORY_DEMUX", params={}),
                ]
            )
        else:
            # Single simulation pipeline
            steps.append(
                TransformStep(
                    name="PRODUCTION_SIMULATION", params={"steps": self.steps}
                )
            )

        # Analysis steps
        steps.extend(
            [
                TransformStep(name="TRAJECTORY_ANALYSIS", params={}),
                TransformStep(
                    name="MSM_BUILD",
                    params={
                        "n_states": self.n_states,
                        "output_dir": str(self.output_dir),
                    },
                ),
                TransformStep(name="BUILD_ANALYSIS", params={}),
            ]
        )

        return TransformPlan(steps=tuple(steps))

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
        if self.prepared_pdb is None:
            raise ValueError("prepare_protein must run before replica exchange setup.")

        config = RemdConfig(
            pdb_file=str(self.prepared_pdb) if self.prepared_pdb else None,
            temperatures=self.temperatures,
            output_dir=str(remd_output_dir),
        )

        self.replica_exchange = ReplicaExchange.from_config(config)
        self.replica_exchange.plan_reporter_stride(
            total_steps=self.steps,
            equilibration_steps=0,
            target_frames=config.target_frames_per_replica,
        )
        self.replica_exchange.setup_replicas()

        logger.info(f"Replica exchange setup for {len(self.temperatures)} temperatures")
        return self.replica_exchange

    def setup_simulation(self) -> Simulation:
        """
        Setup single simulation.

        Returns:
            Simulation object
        """
        logger.info("Stage 2/4: Single Simulation Setup")

        sim_output_dir = self.output_dir / "simulation"
        sim_output_dir.mkdir(parents=True, exist_ok=True)

        self.simulation = Simulation(
            pdb_file=str(self.prepared_pdb),
            output_dir=str(sim_output_dir),
            temperature=self.temperatures[0] if self.temperatures else 300.0,
        )

        logger.info(f"Simulation setup at {self.temperatures[0]}K")
        return self.simulation

    def setup_msm_analysis(self) -> EnhancedMSMProtocol:
        """
        Setup Markov state model analysis.

        Returns:
            MarkovStateModel object
        """
        logger.info("Stage 4/4: MSM Analysis Setup")

        msm_output_dir = self.output_dir / "msm_analysis"
        msm_output_dir.mkdir(parents=True, exist_ok=True)

        self.markov_state_model = MarkovStateModel(output_dir=str(msm_output_dir))

        logger.info(f"MSM setup for {self.n_states} states")
        return self.markov_state_model

    def run(self) -> Dict[str, Any]:
        """
        Run the complete PMARLO pipeline using transform runner.

        Returns:
            Dictionary containing results and output paths
        """
        logger.info("=" * 60)
        logger.info("STARTING PMARLO PIPELINE")
        logger.info("=" * 60)

        # Build the transform plan
        plan = self._build_transform_plan()

        # Initial context with pipeline configuration
        initial_context = {
            "pdb_file": self.pdb_file,
            "temperatures": self.temperatures,
            "steps": self.steps,
            "n_states": self.n_states,
            "output_dir": str(self.output_dir),
            "use_replica_exchange": self.use_replica_exchange,
            "use_metadynamics": self.use_metadynamics,
        }

        try:
            # Run the pipeline using transform runner with optional checkpointing
            checkpoint_dir = str(self.output_dir) if self.enable_checkpoints else None

            final_context = apply_plan(
                plan=plan,
                data=initial_context,
                checkpoint_dir=checkpoint_dir,
                run_id=self.checkpoint_id,
            )

            # Extract results from final context
            results = self._extract_results_from_context(final_context)

            logger.info("=" * 60)
            logger.info("PMARLO PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.info("=" * 60)
            logger.info("PMARLO PIPELINE FAILED")
            logger.info("=" * 60)
            raise

    def _extract_results_from_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pipeline results from the final context."""
        results = {}

        # Protein results
        if "protein" in context and "prepared_pdb" in context:
            protein = context["protein"]
            results["protein"] = {
                "prepared_pdb": str(context["prepared_pdb"]),
                "properties": (
                    protein.get_properties()
                    if hasattr(protein, "get_properties")
                    else {}
                ),
            }

        # Simulation results
        if context.get("use_replica_exchange") and "trajectory_files" in context:
            results["replica_exchange"] = {
                "trajectory_files": context["trajectory_files"],
                "temperatures": [str(t) for t in context.get("temperatures", [])],
                "output_dir": str(Path(context["output_dir"]) / "replica_exchange"),
            }
        elif "trajectory_files" in context:
            results["simulation"] = {
                "trajectory_files": context["trajectory_files"],
                "output_dir": str(Path(context["output_dir"]) / "simulation"),
            }

        # MSM results
        if "msm_result" in context:
            results["msm"] = {
                "output_dir": str(Path(context["output_dir"]) / "msm_analysis"),
                "n_states": str(context.get("n_states", "unknown")),
                "results": context["msm_result"],
            }

        return results

    # ---- Utility methods for CLI and status ----

    def list_runs(self, output_base_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available transform runs with their status."""
        from .runner import TransformManifest

        base_dir = Path(output_base_dir or self.output_dir)
        if not base_dir.exists():
            return []

        runs = []
        for item in base_dir.iterdir():
            if item.is_dir():
                manifest_file = item / ".pmarlo_transform_run.json"
                if manifest_file.exists():
                    try:
                        manifest = TransformManifest(item)
                        manifest.load()
                        run_info = {
                            "run_id": manifest.data.get("run_id", item.name),
                            "status": manifest.data.get("status", "unknown"),
                            "started_at": manifest.data.get("started_at"),
                            "path": str(item),
                        }
                        runs.append(run_info)
                    except Exception as e:
                        logger.warning(f"Failed to load manifest from {item}: {e}")

        return runs


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
        checkpoint_id: Optional checkpoint ID for resuming runs
        auto_continue: Whether to automatically continue interrupted runs
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
