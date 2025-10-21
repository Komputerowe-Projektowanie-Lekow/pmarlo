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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

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


class _ProgressReporter:
    def __init__(
        self,
        *,
        overrides: Mapping[str, str],
        stage_order: Sequence[str],
        stage_map: Mapping[str, str],
        stage_last_step: Mapping[str, str],
        stage_duration_totals: Dict[str, float],
        step_duration_totals: Dict[str, float],
        format_step_name: Callable[[Mapping[str, str], str], str],
    ) -> None:
        self._overrides = overrides
        self._stage_map = stage_map
        self._stage_last_step = stage_last_step
        self._stage_duration_totals = stage_duration_totals
        self._step_duration_totals = step_duration_totals
        self._format_step_name = format_step_name
        self._stage_indices = {label: idx + 1 for idx, label in enumerate(stage_order)}
        self._stage_total = len(stage_order)
        self._current_stage: str | None = None

    def __call__(self, event: str, payload: Mapping[str, Any]) -> None:
        step_name = self._extract_step_name(payload)
        if step_name is None:
            return
        label = self._stage_map.get(step_name)
        if label is None:
            self._report_untracked(event, step_name, payload)
            return
        self._report_stage(event, step_name, label, payload)

    def _extract_step_name(self, payload: Mapping[str, Any]) -> str | None:
        step_name = payload.get("step_name")
        if isinstance(step_name, str):
            return step_name
        return None

    def _report_untracked(
        self, event: str, step_name: str, payload: Mapping[str, Any]
    ) -> None:
        step_display = self._format_step_name(self._overrides, step_name)
        if event == "aggregate_step_start":
            print(f"{step_display}...", flush=True)
            logger.info(f"Entering transform step: {step_display}")
            return
        if event == "aggregate_step_end":
            self._finalise_step(step_name, payload, step_display)

    def _report_stage(
        self,
        event: str,
        step_name: str,
        label: str,
        payload: Mapping[str, Any],
    ) -> None:
        step_display = self._format_step_name(self._overrides, step_name)
        stage_display = self._stage_prefix(label)
        if event == "aggregate_step_start":
            self._handle_stage_start(label)
            message = (
                f"{stage_display} - {step_display}" if stage_display else step_display
            )
            print(f"{message}...", flush=True)
            logger.info(f"Entering transform step: {message}")
            return
        if event == "aggregate_step_end":
            self._finalise_stage_step(label, step_name, payload, step_display)

    def _handle_stage_start(self, label: str) -> None:
        if label == self._current_stage:
            return
        announce_stage_start(
            label,
            logger=logger,
            index=self._stage_indices.get(label),
            total=self._stage_total,
        )
        self._current_stage = label

    def _finalise_step(
        self, step_name: str, payload: Mapping[str, Any], step_display: str
    ) -> None:
        duration = payload.get("duration_s")
        message = f"{step_display} complete"
        if isinstance(duration, (int, float)):
            duration_value = float(duration)
            self._step_duration_totals[step_name] = duration_value
            message += f" ({format_duration(duration_value)})"
        print(message, flush=True)
        logger.info(message)

    def _finalise_stage_step(
        self,
        label: str,
        step_name: str,
        payload: Mapping[str, Any],
        step_display: str,
    ) -> None:
        duration = payload.get("duration_s")
        message = f"{step_display} complete"
        if isinstance(duration, (int, float)):
            duration_value = float(duration)
            self._step_duration_totals[step_name] = duration_value
            self._stage_duration_totals[label] += duration_value
            message += f" ({format_duration(duration_value)})"
        print(message, flush=True)
        logger.info(message)
        if self._stage_last_step.get(label) != step_name:
            return
        stage_elapsed = self._stage_duration_totals.get(label, 0.0)
        details: list[str] = []
        if stage_elapsed > 0.0:
            details.append(f"Duration: {format_duration(stage_elapsed)}")
        announce_stage_complete(
            label,
            logger=logger,
            details=details or None,
        )

    def _stage_prefix(self, label: str) -> str:
        index = self._stage_indices.get(label)
        if index is None or self._stage_total == 0:
            return label
        return f"Stage {index}/{self._stage_total}: {label}"


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

        print("PMARLO Pipeline configuration:", flush=True)
        print(f"  Input PDB: {self.pdb_file}", flush=True)
        print(f"  Output directory: {self.output_dir}", flush=True)
        print(f"  Steps: {self.steps}", flush=True)
        print(f"  Replica exchange enabled: {self.use_replica_exchange}", flush=True)
        print(f"  Metadynamics enabled: {self.use_metadynamics}", flush=True)
        if self.use_replica_exchange:
            replicas = len(self.temperatures)
            temp_min = min(self.temperatures)
            temp_max = max(self.temperatures)
            print(f"  Replicas: {replicas}", flush=True)
            print(
                f"  Temperature range: {temp_min:.1f}K - {temp_max:.1f}K",
                flush=True,
            )
        else:
            print(f"  Temperature: {self.temperatures[0]:.1f}K", flush=True)
        print(f"  Checkpoints enabled: {self.enable_checkpoints}", flush=True)

        if self.use_replica_exchange:
            stage_sequence: tuple[str, ...] = (
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
        steps: list[TransformStep] = []

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
        return str(format_stage_header(label, index=index, total=total))

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

    def _stage_definitions(self) -> list[tuple[str, set[str]]]:
        if self.use_replica_exchange:
            return [
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
        return [
            ("Protein Preparation", {"PROTEIN_PREPARATION"}),
            ("Simulation", {"PRODUCTION_SIMULATION"}),
            (
                "MSM Analysis Setup",
                {"TRAJECTORY_ANALYSIS", "MSM_BUILD", "BUILD_ANALYSIS"},
            ),
        ]

    @staticmethod
    def _step_name_overrides() -> dict[str, str]:
        return {
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

    @staticmethod
    def _resolve_stage_label(
        stage_definitions: list[tuple[str, set[str]]], step_name: str
    ) -> str:
        for label, names in stage_definitions:
            if step_name in names:
                return label
        return step_name.replace("_", " ").title()

    @staticmethod
    def _format_step_name(overrides: Mapping[str, str], step_name: str) -> str:
        if step_name in overrides:
            return overrides[step_name]
        tokens = step_name.split("_")
        acronyms = {"MSM", "REMD", "CV"}
        return " ".join(
            token if token.upper() in acronyms else token.capitalize()
            for token in tokens
        )

    def _prepare_stage_context(
        self, plan, stage_definitions: list[tuple[str, set[str]]]
    ) -> tuple[dict[str, str], dict[str, str], list[str], dict[str, int]]:
        stage_map: dict[str, str] = {}
        stage_last_step: dict[str, str] = {}
        stage_order = [label for label, _ in stage_definitions]
        for step in plan.steps:
            label = self._resolve_stage_label(stage_definitions, step.name)
            stage_map[step.name] = label
            stage_last_step[label] = step.name
        stage_indices = {label: idx + 1 for idx, label in enumerate(stage_order)}
        return stage_map, stage_last_step, stage_order, stage_indices

    def _make_progress_handler(
        self,
        _stage_definitions: list[tuple[str, set[str]]],
        stage_order: list[str],
        stage_map: dict[str, str],
        stage_last_step: dict[str, str],
        stage_duration_totals: Dict[str, float],
        step_duration_totals: Dict[str, float],
    ) -> Callable[[str, Mapping[str, Any]], None]:
        overrides = self._step_name_overrides()
        reporter = _ProgressReporter(
            overrides=overrides,
            stage_order=stage_order,
            stage_map=stage_map,
            stage_last_step=stage_last_step,
            stage_duration_totals=stage_duration_totals,
            step_duration_totals=step_duration_totals,
            format_step_name=self._format_step_name,
        )
        return reporter

    @staticmethod
    def _start_tracemalloc() -> tuple[Any | None, bool]:
        try:
            import tracemalloc as _tracemalloc  # type: ignore

            if not _tracemalloc.is_tracing():
                _tracemalloc.start()
                return _tracemalloc, True
            return _tracemalloc, False
        except ImportError:  # pragma: no cover - optional tooling
            return None, False

    @staticmethod
    def _stop_tracemalloc(tracemalloc_mod: Any | None, started: bool) -> None:
        if tracemalloc_mod is None:
            return
        if started and tracemalloc_mod.is_tracing():
            tracemalloc_mod.stop()

    def _summarize_stage_timings(
        self, stage_order: Sequence[str], stage_duration_totals: Mapping[str, float]
    ) -> None:
        if not stage_duration_totals:
            return
        print("Stage timing summary:", flush=True)
        logger.info("Stage timing summary:")
        for label in stage_order:
            duration_value = stage_duration_totals.get(label, 0.0)
            if duration_value <= 0.0:
                continue
            summary_line = f"- {label}: {format_duration(duration_value)}"
            print(summary_line, flush=True)
            logger.info(summary_line)

    def _summarize_results(self, results: Mapping[str, Any]) -> None:
        replica_info = results.get("replica_exchange")
        if replica_info:
            traj_files = replica_info.get("trajectory_files", [])
            out_dir = replica_info.get("output_dir")
            summary = (
                f"Replica exchange produced {len(traj_files)} trajectories -> {out_dir}"
            )
            print(summary, flush=True)
            logger.info(summary)

        sim_info = results.get("simulation")
        if sim_info:
            traj_files = sim_info.get("trajectory_files", [])
            out_dir = sim_info.get("output_dir")
            summary = f"Simulation produced {len(traj_files)} trajectories -> {out_dir}"
            print(summary, flush=True)
            logger.info(summary)

        msm_info = results.get("msm")
        if msm_info:
            n_states = msm_info.get("n_states", "unknown")
            out_dir = msm_info.get("output_dir")
            summary = f"MSM analysis available with {n_states} states -> {out_dir}"
            print(summary, flush=True)
            logger.info(summary)

    def _report_throughput(self, total_elapsed: float) -> None:
        if total_elapsed <= 0.0:
            return
        md_steps = int(max(0, self.steps))
        if md_steps <= 0:
            return
        steps_per_sec = md_steps / total_elapsed
        if self.use_replica_exchange:
            replicas = max(1, len(self.temperatures))
            throughput_msg = (
                f"Simulation throughput: {md_steps} steps across {replicas} replicas "
                f"(~{steps_per_sec:.1f} steps/s per replica)"
            )
        else:
            throughput_msg = f"Simulation throughput: {md_steps} steps (~{steps_per_sec:.1f} steps/s)"
        print(throughput_msg, flush=True)
        logger.info(throughput_msg)

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

        stage_duration_totals: Dict[str, float] = defaultdict(float)
        step_duration_totals: Dict[str, float] = {}
        stage_definitions = self._stage_definitions()
        stage_map, stage_last_step, stage_order, _ = self._prepare_stage_context(
            plan, stage_definitions
        )
        progress = self._make_progress_handler(
            stage_definitions,
            stage_order,
            stage_map,
            stage_last_step,
            stage_duration_totals,
            step_duration_totals,
        )

        pipeline_start = perf_counter()
        tracemalloc_mod, tracemalloc_started = self._start_tracemalloc()

        try:
            # Run the pipeline using transform runner with optional checkpointing
            checkpoint_dir = str(self.output_dir) if self.enable_checkpoints else None

            final_context = apply_plan(
                plan=plan,
                data=initial_context,
                progress_callback=progress,
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

            self._report_throughput(total_elapsed)
            self._summarize_stage_timings(stage_order, stage_duration_totals)

            peak_memory_mb: float | None = None
            if tracemalloc_mod is not None and tracemalloc_mod.is_tracing():
                _current, peak_bytes = tracemalloc_mod.get_traced_memory()
                peak_memory_mb = peak_bytes / (1024.0 * 1024.0)
            self._stop_tracemalloc(tracemalloc_mod, tracemalloc_started)
            if peak_memory_mb is not None and peak_memory_mb > 0.0:
                mem_msg = f"Peak memory usage (tracked): {peak_memory_mb:.1f} MB"
                logger.info(mem_msg)

            self._summarize_results(results)
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            print("ERROR: Pipeline failed, see logs for details.", flush=True)
            self._stop_tracemalloc(tracemalloc_mod, tracemalloc_started)
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
