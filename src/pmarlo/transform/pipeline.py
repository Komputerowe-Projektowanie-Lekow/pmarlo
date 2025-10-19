# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Pipeline orchestration module for PMARLO.

Provides a simple interface to coordinate protein preparation, replica exchange,
simulation, and Markov state model analysis using the transform runner system.
"""

import logging
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

from pmarlo.utils.logging_utils import (
    StageTimer,
    announce_stage_complete,
    announce_stage_start,
    emit_banner,
    format_duration,
    format_stage_header,
)
from pmarlo.utils.path_utils import ensure_directory

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

        self._stage_durations: Dict[str, float] = {}

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
        ensure_directory(self.output_dir)

        logger.info("PMARLO Pipeline initialized")
        logger.info(f"  PDB file: {self.pdb_file}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Temperatures: {self.temperatures}")
        logger.info(f"  Replica Exchange: {self.use_replica_exchange}")
        logger.info(f"  Metadynamics: {self.use_metadynamics}")
        logger.info(f"  Checkpoints enabled: {self.enable_checkpoints}")

        if self.use_replica_exchange:
            stage_sequence = (
                "Protein Preparation",
                "System Setup",
                "Simulation",
                "MSM Analysis Setup",
            )
        else:
            stage_sequence = (
                "Protein Preparation",
                "Simulation",
                "MSM Analysis Setup",
            )
        self._stage_sequence: tuple[str, ...] = stage_sequence
        self._stage_index_map = {
            label: idx + 1 for idx, label in enumerate(self._stage_sequence)
        }
        self._stage_total = len(self._stage_sequence)

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

    def _stage_header(self, label: str) -> str:
        index = self._stage_index_map.get(label)
        total = self._stage_total if index is not None else None
        return format_stage_header(label, index=index, total=total)

    def setup_protein(self, ph: float = 7.0) -> Protein:
        """
        Setup and prepare the protein.

        Args:
            ph: pH for protonation state

        Returns:
            Prepared Protein object
        """
        header = self._stage_header("Protein Preparation")
        print(f"{header}...", flush=True)
        logger.info(header)

        with StageTimer("Protein preparation", logger=logger) as timer:
            self.protein = Protein(self.pdb_file, ph=ph)

            # Save prepared protein
            self.prepared_pdb = self.output_dir / "prepared_protein.pdb"
            self.protein.save(str(self.prepared_pdb))

            properties = self.protein.get_properties()
            atom_count = (
                properties.get("num_atoms", "unknown") if properties else "unknown"
            )
            residue_count = (
                properties.get("num_residues", "unknown") if properties else "unknown"
            )
            summary = (
                "Protein prepared: "
                f"{atom_count} atoms, "
                f"{residue_count} residues -> {self.prepared_pdb}"
            )
            print(summary, flush=True)
            logger.info(summary)

        self._stage_durations["Protein Preparation"] = timer.elapsed
        return self.protein

    def setup_replica_exchange(self) -> Optional[ReplicaExchange]:
        """
        Setup replica exchange if enabled.

        Returns:
            ReplicaExchange object if enabled, None otherwise
        """
        if not self.use_replica_exchange:
            return None

        header = self._stage_header("System Setup")
        print(f"{header}...", flush=True)
        logger.info(header)

        remd_output_dir = self.output_dir / "replica_exchange"
        if self.prepared_pdb is None:
            raise ValueError("prepare_protein must run before replica exchange setup.")

        config = RemdConfig(
            pdb_file=str(self.prepared_pdb) if self.prepared_pdb else None,
            temperatures=self.temperatures,
            output_dir=str(remd_output_dir),
        )

        with StageTimer("Replica exchange setup", logger=logger) as timer:
            self.replica_exchange = ReplicaExchange.from_config(config)
            self.replica_exchange.plan_reporter_stride(
                total_steps=self.steps,
                equilibration_steps=0,
                target_frames=config.target_frames_per_replica,
            )
            self.replica_exchange.setup_replicas()

            summary = (
                "Replica exchange configured: "
                f"{len(self.temperatures)} replicas, "
                f"output -> {remd_output_dir}"
            )
            print(summary, flush=True)
            logger.info(summary)

        self._stage_durations["System Setup"] = timer.elapsed
        return self.replica_exchange

    def setup_simulation(self) -> Simulation:
        """
        Setup single simulation.

        Returns:
            Simulation object
        """
        header = self._stage_header("Simulation")
        print(f"{header}...", flush=True)
        logger.info(header)

        sim_output_dir = self.output_dir / "simulation"
        ensure_directory(sim_output_dir)

        with StageTimer("Simulation setup", logger=logger) as timer:
            self.simulation = Simulation(
                pdb_file=str(self.prepared_pdb),
                output_dir=str(sim_output_dir),
                temperature=self.temperatures[0] if self.temperatures else 300.0,
            )

            temp = self.temperatures[0] if self.temperatures else 300.0
            summary = f"Simulation configured at {temp:.1f} K -> {sim_output_dir}"
            print(summary, flush=True)
            logger.info(summary)

        self._stage_durations["Simulation Setup"] = timer.elapsed
        return self.simulation

    def setup_msm_analysis(self) -> EnhancedMSMProtocol:
        """
        Setup Markov state model analysis.

        Returns:
            MarkovStateModel object
        """
        header = self._stage_header("MSM Analysis Setup")
        print(f"{header}...", flush=True)
        logger.info(header)

        msm_output_dir = self.output_dir / "msm_analysis"
        ensure_directory(msm_output_dir)

        with StageTimer("MSM analysis setup", logger=logger) as timer:
            self.markov_state_model = MarkovStateModel(output_dir=str(msm_output_dir))

            summary = f"MSM analysis initialised for {self.n_states} states -> {msm_output_dir}"
            print(summary, flush=True)
            logger.info(summary)

        self._stage_durations["MSM Analysis Setup"] = timer.elapsed
        return self.markov_state_model

    def get_components(self) -> Dict[str, Any]:
        """Return currently initialised pipeline components."""
        return {
            "protein": self.protein,
            "replica_exchange": self.replica_exchange,
            "simulation": self.simulation,
            "markov_state_model": self.markov_state_model,
        }

    def run(self) -> Dict[str, Any]:
        """
        Run the complete PMARLO pipeline using transform runner.

        Returns:
            Dictionary containing results and output paths
        """
        emit_banner(
            "PMARLO PIPELINE START",
            logger=logger,
            details=[
                f"PDB file: {self.pdb_file}",
                f"Output directory: {self.output_dir}",
                f"Replica exchange enabled: {self.use_replica_exchange}",
                f"Metadynamics enabled: {self.use_metadynamics}",
            ],
        )

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

        stage_map: Dict[str, str] = {}
        stage_last_step: Dict[str, str] = {}

        if self.use_replica_exchange:
            stage_definitions: list[tuple[str, set[str]]] = [
                ("Protein Preparation", {"PROTEIN_PREPARATION"}),
                (
                    "System Setup",
                    {"SYSTEM_SETUP", "REPLICA_INITIALIZATION", "ENERGY_MINIMIZATION"},
                ),
                (
                    "Simulation",
                    {
                        "GRADUAL_HEATING",
                        "EQUILIBRATION",
                        "PRODUCTION_SIMULATION",
                        "TRAJECTORY_DEMUX",
                    },
                ),
                (
                    "MSM Analysis Setup",
                    {"TRAJECTORY_ANALYSIS", "MSM_BUILD", "BUILD_ANALYSIS"},
                ),
            ]
        else:
            stage_definitions = [
                ("Protein Preparation", {"PROTEIN_PREPARATION"}),
                ("Simulation", {"PRODUCTION_SIMULATION"}),
                (
                    "MSM Analysis Setup",
                    {"TRAJECTORY_ANALYSIS", "MSM_BUILD", "BUILD_ANALYSIS"},
                ),
            ]

        stage_order = [label for label, _ in stage_definitions]

        def _resolve_stage_label(step_name: str) -> str:
            for label, names in stage_definitions:
                if step_name in names:
                    return label
            return step_name.replace("_", " ").title()

        step_name_overrides = {
            "PROTEIN_PREPARATION": "Protein Preparation",
            "SYSTEM_SETUP": "System Setup",
            "REPLICA_INITIALIZATION": "Replica Initialization",
            "ENERGY_MINIMIZATION": "Energy Minimization",
            "GRADUAL_HEATING": "Gradual Heating",
            "EQUILIBRATION": "Equilibration",
            "PRODUCTION_SIMULATION": "Production Simulation",
            "TRAJECTORY_DEMUX": "Trajectory Demultiplexing",
            "TRAJECTORY_ANALYSIS": "Trajectory Analysis",
            "MSM_BUILD": "MSM Analysis Setup",
            "BUILD_ANALYSIS": "Analysis Artifact Build",
        }

        def _format_step_name(step_name: str) -> str:
            if step_name in step_name_overrides:
                return step_name_overrides[step_name]
            tokens = step_name.split("_")
            acronyms = {"MSM", "REMD", "CV"}
            return " ".join(
                token if token.upper() in acronyms else token.capitalize()
                for token in tokens
            )

        for step in plan.steps:
            label = _resolve_stage_label(step.name)
            stage_map[step.name] = label
            stage_last_step[label] = step.name

        stage_indices = {label: idx + 1 for idx, label in enumerate(stage_order)}
        stage_total = len(stage_order)
        current_stage: str | None = None
        stage_duration_totals: Dict[str, float] = defaultdict(float)
        step_duration_totals: Dict[str, float] = {}

        def _stage_prefix(label: str | None) -> str:
            if label is None:
                return ""
            index = stage_indices.get(label)
            if index is None:
                return label
            return f"Stage {index}/{stage_total}: {label}"

        def _progress(event: str, payload: Dict[str, Any]) -> None:
            nonlocal current_stage
            step_name = payload.get("step_name")
            if not isinstance(step_name, str):
                return
            label = stage_map.get(step_name)
            if label is None:
                step_display = _format_step_name(step_name)
                if event == "aggregate_step_start":
                    print(f"{step_display}...", flush=True)
                    logger.info(f"Entering transform step: {step_display}")
                elif event == "aggregate_step_end":
                    duration = payload.get("duration_s")
                    message = f"{step_display} complete"
                    if isinstance(duration, (int, float)):
                        step_duration_totals[step_name] = float(duration)
                        message += f" ({format_duration(float(duration))})"
                    print(message, flush=True)
                    logger.info(message)
                return
            step_display = _format_step_name(step_name)
            stage_display = _stage_prefix(label)
            if event == "aggregate_step_start":
                if label != current_stage:
                    announce_stage_start(
                        label,
                        logger=logger,
                        index=stage_indices.get(label),
                        total=stage_total,
                    )
                    current_stage = label
                message = (
                    f"{stage_display} - {step_display}"
                    if stage_display
                    else step_display
                )
                print(f"{message}...", flush=True)
                logger.info(f"Entering transform step: {message}")
            elif event == "aggregate_step_end":
                duration = payload.get("duration_s")
                message = f"{step_display} complete"
                if isinstance(duration, (int, float)):
                    step_duration_totals[step_name] = float(duration)
                    stage_duration_totals[label] += float(duration)
                    message += f" ({format_duration(float(duration))})"
                print(message, flush=True)
                logger.info(message)
                if stage_last_step.get(label) == step_name:
                    stage_elapsed = stage_duration_totals.get(label, 0.0)
                    details_lines = []
                    if stage_elapsed > 0.0:
                        details_lines.append(
                            f"Duration: {format_duration(stage_elapsed)}"
                        )
                    announce_stage_complete(
                        label,
                        logger=logger,
                        details=details_lines or None,
                    )

        pipeline_start = perf_counter()
        tracemalloc_mod = None
        tracemalloc_started = False
        try:
            import tracemalloc as _tracemalloc  # type: ignore

            tracemalloc_mod = _tracemalloc
            if not tracemalloc_mod.is_tracing():
                tracemalloc_mod.start()
                tracemalloc_started = True
        except ImportError:  # pragma: no cover - optional tooling
            tracemalloc_mod = None

        try:
            # Run the pipeline using transform runner with optional checkpointing
            checkpoint_dir = str(self.output_dir) if self.enable_checkpoints else None

            final_context = apply_plan(
                plan=plan,
                data=initial_context,
                progress_callback=_progress,
                checkpoint_dir=checkpoint_dir,
                run_id=self.checkpoint_id,
            )

            # Extract results from final context
            results = self._extract_results_from_context(final_context)

            emit_banner(
                format_stage_header("PMARLO PIPELINE COMPLETE"),
                logger=logger,
                details=["All workflow stages completed successfully."],
            )

            total_elapsed = perf_counter() - pipeline_start
            total_elapsed_msg = (
                f"Pipeline completed in {format_duration(total_elapsed)}"
            )
            print(total_elapsed_msg, flush=True)
            logger.info(total_elapsed_msg)

            if total_elapsed > 0.0:
                md_steps = int(max(0, self.steps))
                if md_steps > 0:
                    steps_per_sec = md_steps / total_elapsed
                    if self.use_replica_exchange:
                        replicas = max(1, len(self.temperatures))
                        throughput_msg = (
                            f"Simulation throughput: {md_steps} steps across "
                            f"{replicas} replicas (~{steps_per_sec:.1f} steps/s per replica)"
                        )
                    else:
                        throughput_msg = (
                            f"Simulation throughput: {md_steps} steps "
                            f"(~{steps_per_sec:.1f} steps/s)"
                        )
                    print(throughput_msg, flush=True)
                    logger.info(throughput_msg)

            if stage_duration_totals:
                print("Stage timing summary:", flush=True)
                logger.info("Stage timing summary:")
                for label in stage_order:
                    duration_value = stage_duration_totals.get(label, 0.0)
                    if duration_value <= 0.0:
                        continue
                    summary_line = f"- {label}: {format_duration(duration_value)}"
                    print(summary_line, flush=True)
                    logger.info(summary_line)

            peak_memory_mb: float | None = None
            if tracemalloc_mod is not None and tracemalloc_mod.is_tracing():
                _current, peak_bytes = tracemalloc_mod.get_traced_memory()
                peak_memory_mb = peak_bytes / (1024.0 * 1024.0)
                if tracemalloc_started:
                    tracemalloc_mod.stop()
            if peak_memory_mb is not None and peak_memory_mb > 0.0:
                mem_msg = f"Peak memory usage (tracked): {peak_memory_mb:.1f} MB"
                logger.info(mem_msg)

            replica_info = results.get("replica_exchange")
            if replica_info:
                traj_files = replica_info.get("trajectory_files", [])
                out_dir = replica_info.get("output_dir")
                summary = f"Replica exchange produced {len(traj_files)} trajectories -> {out_dir}"
                print(summary, flush=True)
                logger.info(summary)

            sim_info = results.get("simulation")
            if sim_info:
                traj_files = sim_info.get("trajectory_files", [])
                out_dir = sim_info.get("output_dir")
                summary = (
                    f"Simulation produced {len(traj_files)} trajectories -> {out_dir}"
                )
                print(summary, flush=True)
                logger.info(summary)

            msm_info = results.get("msm")
            if msm_info:
                n_states = msm_info.get("n_states", "unknown")
                out_dir = msm_info.get("output_dir")
                summary = f"MSM analysis available with {n_states} states -> {out_dir}"
                print(summary, flush=True)
                logger.info(summary)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            if (
                tracemalloc_mod is not None
                and tracemalloc_started
                and tracemalloc_mod.is_tracing()
            ):
                tracemalloc_mod.stop()
            emit_banner(
                format_stage_header("PMARLO PIPELINE FAILED"),
                logger=logger,
                details=[f"Reason: {e}"],
            )
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
