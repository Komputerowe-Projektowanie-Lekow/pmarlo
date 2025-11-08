# Standard library imports
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Third-party imports
import mdtraj as md
import numpy as np

# pmarlo imports
from pmarlo.api.replica_exchange import run_replica_exchange, _derive_run_plan
from pmarlo.utils.path_utils import ensure_directory

# Local imports
from .run_metadata import load_run_plan, save_run_plan
from .types import SimulationConfig, SimulationResult
from .utils import _coerce_path_list, _slugify, _timestamp
from .layout import WorkspaceLayout

logger = logging.getLogger(__name__)

DEFAULT_SAVE_STATE_FREQUENCY = 10_000


# Module-level helper functions (can be imported by other modules and frontend tabs)
def run_short_sim(
    pdb_path: Path,
    workspace: Path,
    temperatures: Sequence[float],
    *,
    steps: int = 1000,
    quick: bool = True,
    random_seed: Optional[int] = None,
    start_from: Optional[Path] = None,
    use_stub: Optional[bool] = None,
) -> SimulationResult:
    """Run a short simulation for testing purposes.

    Parameters
    ----------
    pdb_path:
        Path to the input PDB file.
    workspace:
        Root workspace directory.
    temperatures:
        Temperature ladder for REMD.
    steps:
        Number of simulation steps (default 1000).
    quick:
        Whether to use quick mode (reduced steps, default True).
    random_seed:
        Random seed for reproducibility.
    start_from:
        Optional restart PDB to continue from.
    use_stub:
        When ``True`` (default if ``quick`` is ``True``), generate synthetic
        trajectories rather than invoking the full REMD stack. Setting this to
        ``False`` forces a real simulation even in quick mode.

    Returns
    -------
    SimulationResult
        The result of the simulation run.
    """
    # Import WorkflowBackend here to avoid circular imports
    from .workspace import WorkflowBackend

    layout = WorkspaceLayout(
        app_root=workspace,
        inputs_dir=workspace / "inputs",
        workspace_dir=workspace / "output",
        sims_dir=workspace / "output" / "sims",
        shards_dir=workspace / "output" / "shards",
        models_dir=workspace / "output" / "models",
        bundles_dir=workspace / "output" / "bundles",
        logs_dir=workspace / "output" / "logs",
        state_path=workspace / "output" / "state.json",
    )
    layout.ensure()

    backend = WorkflowBackend(layout)
    effective_steps = int(steps)
    if quick:
        effective_steps = max(1, min(effective_steps, 200))

    stub_result = quick if use_stub is None else bool(use_stub)

    config = SimulationConfig(
        pdb_path=pdb_path,
        temperatures=temperatures,
        steps=effective_steps,
        quick=quick,
        random_seed=random_seed,
        stub_result=stub_result,
        start_from_pdb=start_from,
    )
    return backend.run_sampling(config)


def _resolve_save_state_frequency(config: SimulationConfig) -> int:
    return int(config.save_state_frequency or DEFAULT_SAVE_STATE_FREQUENCY)


def _resolve_primary_temperature(config: SimulationConfig) -> float:
    """Return the primary simulation temperature."""

    if config.temperatures:
        return float(config.temperatures[0])
    if config.restart_temperature is not None:
        return float(config.restart_temperature)
    return 300.0


def _derive_characteristics(config: SimulationConfig) -> Tuple[int, int, int]:
    equilibration_steps, exchange_frequency, dcd_stride = _derive_run_plan(
        int(config.steps),
        bool(config.quick),
        int(config.exchange_frequency_steps)
        if config.exchange_frequency_steps is not None
        else None,
    )
    return equilibration_steps, exchange_frequency, dcd_stride


def _normalize_resume_context(ctx: Dict[str, Any], *, timestamp: str) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in ctx.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        elif isinstance(value, (list, tuple)):
            normalized[key] = [
                str(item) if isinstance(item, Path) else item for item in value
            ]
        else:
            normalized[key] = value
    normalized.setdefault("timestamp", timestamp)
    return normalized


# SamplingMixin class containing all methods for sampling operations
class SamplingMixin:
    """Methods for REMD sampling and simulation management.

    This class is mixed into the Backend class to provide simulation
    orchestration, restart handling, and stub generation functionality.
    """

    def load_run(self, run_id: str) -> Optional[SimulationResult]:
        """Best-effort reconstruction of a previous simulation result.

        Parameters
        ----------
        run_id:
            The unique identifier for the simulation run.

        Returns
        -------
        Optional[SimulationResult]
            The reconstructed simulation result, or None if not found.
        """
        if not run_id:
            return None
        record: Optional[Dict[str, Any]] = None
        for entry in reversed(self.state.runs):
            if str(entry.get("run_id")) == str(run_id):
                record = dict(entry)
                break
        if record is None:
            return None

        run_dir = self._path_from_value(record.get("run_dir"))
        pdb_path = self._path_from_value(record.get("pdb"))
        if run_dir is None or pdb_path is None:
            return None
        if not run_dir.exists() or not pdb_path.exists():
            return None

        traj_entries = record.get("traj_files", []) or []
        traj_files: List[Path] = []
        for entry in traj_entries:
            path = self._path_from_value(entry)
            if path is not None and path.exists():
                traj_files.append(path)
        if not traj_files:
            raise FileNotFoundError(
                f"No trajectory files found for run {run_dir}. "
                f"Expected trajectory files in analysis results or standard REMD output locations."
            )
        analysis_temps = [float(t) for t in record.get("analysis_temperatures", [])]
        steps = int(record.get("steps", 0))
        created_at = str(record.get("created_at", "")) or _timestamp()
        restart_pdb_path: Optional[Path] = None
        restart_inputs_entry: Optional[Path] = None
        restart_pdb_path = self._path_from_value(record.get("restart_pdb"))
        if restart_pdb_path is not None and not restart_pdb_path.exists():
            restart_pdb_path = None
        restart_inputs_entry = self._path_from_value(record.get("restart_input_entry"))
        if restart_inputs_entry is not None and not restart_inputs_entry.exists():
            restart_inputs_entry = None

        return SimulationResult(
            run_id=str(record.get("run_id")),
            run_dir=run_dir.resolve(),
            pdb_path=pdb_path.resolve(),
            traj_files=[p.resolve() for p in traj_files],
            analysis_temperatures=analysis_temps,
            steps=steps,
            created_at=created_at,
            restart_pdb_path=restart_pdb_path,
            restart_inputs_entry=restart_inputs_entry,
        )

    def delete_simulation(self, index: int) -> bool:
        """Delete a simulation run and its associated files.

        Parameters
        ----------
        index:
            The index of the simulation in the state runs list.

        Returns
        -------
        bool
            True if deletion was successful, False otherwise.
        """
        entry = self.state.remove_run(index)
        if entry is None:
            return False

        try:
            # Delete simulation directory
            run_dir = self._path_from_value(entry.get("run_dir"))
            if run_dir is not None and run_dir.exists() and run_dir.is_dir():
                shutil.rmtree(run_dir)

            # Also remove any associated shards
            run_id = entry.get("run_id", "")
            if run_id:
                # Find and remove associated shard entries
                shards_to_remove = []
                for i, shard_entry in enumerate(self.state.shards):
                    if shard_entry.get("run_id") == run_id:
                        shards_to_remove.append(i)

                # Remove in reverse order to maintain indices
                for i in reversed(shards_to_remove):
                    self.delete_shard_batch(i)

            return True
        except Exception:
            return False

    def _plan_restart_snapshot_paths(
        self,
        *,
        run_label: str,
        run_dir: Path,
        source_pdb: Path,
    ) -> Tuple[Path, Path]:
        """Determine output locations for restart snapshots.

        Parameters
        ----------
        run_label:
            The label for this simulation run.
        run_dir:
            The simulation output directory.
        source_pdb:
            The original input PDB file.

        Returns
        -------
        Tuple[Path, Path]
            A tuple of (run_path, inputs_path) for the restart snapshot.

        Raises
        ------
        FileNotFoundError:
            If the source PDB does not exist.
        FileExistsError:
            If a restart PDB already exists at the planned location.
        """
        source = Path(source_pdb).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Protein input {source} does not exist.")
        filename = f"{source.stem}_{run_label}.pdb"

        snapshot_dir = run_dir / "restart"
        ensure_directory(snapshot_dir)
        ensure_directory(self.layout.inputs_dir)

        run_path = (snapshot_dir / filename).resolve()
        inputs_path = (self.layout.inputs_dir / filename).resolve()

        for candidate in (run_path, inputs_path):
            if candidate.exists():
                raise FileExistsError(
                    f"Restart PDB already exists at {candidate}. Remove it or choose a different run label."
                )
        return run_path, inputs_path

    def run_sampling(self, config: SimulationConfig) -> SimulationResult:
        """Execute a REMD sampling run with the given configuration.

        Parameters
        ----------
        config:
            The simulation configuration.

        Returns
        -------
        SimulationResult
            The result of the simulation run.
        """
        planned_at = _timestamp()
        base_label = _slugify(config.label) or f"run-{planned_at}"

        # Prepare CV model info if provided
        # CV biasing is now properly implemented with harmonic expansion bias
        # The exported model includes CVBiasPotential wrapper that transforms CVs → Energy
        cv_kwargs = {}
        use_stub = bool(config.stub_result)
        if config.cv_model_bundle:
            use_stub = False

            logger.info("=" * 60)
            logger.info("CV-INFORMED SAMPLING ENABLED")
            logger.info("=" * 60)

            from pmarlo.features.deeptica import (
                check_openmm_torch_available,
                load_cv_model_info,
            )

            if not check_openmm_torch_available():
                raise RuntimeError(
                    "CV-informed sampling requested but openmm-torch is not installed."
                )

            try:
                import torch
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "CV-informed sampling requires PyTorch to be installed."
                ) from exc

            if not torch.cuda.is_available():  # pragma: no cover - hardware dependent
                logger.warning(
                    "WARNING: PyTorch is running on CPU only!\n"
                    "CV-biased simulations will be ~10-20x slower than unbiased.\n"
                    "For production use, install CUDA-enabled PyTorch:\n"
                    "https://pytorch.org/get-started/locally/\n"
                )

            bundle_path = Path(config.cv_model_bundle)
            if not bundle_path.exists():
                raise FileNotFoundError(
                    f"CV model bundle {bundle_path} does not exist."
                )

            cv_info = load_cv_model_info(bundle_path, model_name="deeptica_cv_model")
            cv_kwargs["cv_model_path"] = cv_info["model_path"]
            cv_kwargs["cv_scaler_mean"] = cv_info["scaler_params"]["mean"]
            cv_kwargs["cv_scaler_scale"] = cv_info["scaler_params"]["scale"]

            logger.info("OK CV bias potential loaded successfully")
            logger.info(f"  Model path: {cv_info['model_path']}")
            logger.info(f"  CV dimensions: {cv_info['config']['cv_dim']}")
            logger.info(f"  Bias type: {cv_info['config'].get('bias_type', 'harmonic_expansion')}")
            logger.info(f"  Bias strength: {cv_info['config'].get('bias_strength', 10.0):.1f} kJ/mol")
            logger.info("\nBias physics:")
            logger.info("  E_bias = k * sum(cv_i^2)")
            logger.info("  Forces: F = -∇E_bias (computed by OpenMM)")
            logger.info("  Purpose: Repulsive bias → explore diverse conformations")
            logger.info("\nIMPORTANT: The model expects MOLECULAR FEATURES as input")
            logger.info("  (distances, angles, dihedrals), not raw atomic positions.")
            logger.info("  Feature extraction must be configured in OpenMM system.")
        run_label = base_label
        if config.force_run_id:
            run_label = _slugify(config.force_run_id)
        elif config.random_seed is not None:
            run_label = f"{base_label}-seed-{int(config.random_seed)}"
        elif use_stub:
            run_label = f"{base_label}-stub-{len(self.state.runs)}"
        run_dir = (self.layout.sims_dir / run_label).resolve()
        ensure_directory(run_dir)

        restart_paths: Optional[Tuple[Path, Path]] = None
        restart_target_temperature: Optional[float] = None
        if config.save_restart_pdb:
            restart_target_temperature = (
                float(config.restart_temperature)
                if config.restart_temperature is not None
                else _resolve_primary_temperature(config)
            )
            restart_paths = self._plan_restart_snapshot_paths(
                run_label=run_label,
                run_dir=run_dir,
                source_pdb=Path(config.pdb_path),
            )

        save_state_frequency = _resolve_save_state_frequency(config)
        equilibration_steps, exchange_frequency, dcd_stride = _derive_characteristics(
            config
        )

        config_snapshot = config.snapshot()
        config_snapshot["run_id"] = run_label
        config_snapshot["run_dir"] = str(run_dir)

        existing_plan = load_run_plan(run_dir)
        resume_history: List[Dict[str, Any]] = []
        if existing_plan:
            history = existing_plan.get("resume_history")
            if isinstance(history, list):
                resume_history = list(history)

        resume_context_norm: Optional[Dict[str, Any]] = None
        if config.resume_context:
            resume_context_norm = _normalize_resume_context(
                config.resume_context, timestamp=planned_at
            )

        plan_payload: Dict[str, Any] = {
            "schema": "pmarlo.run-plan",
            "version": 1,
            "run_id": run_label,
            "run_dir": str(run_dir),
            "planned_at": planned_at,
            "status": "pending",
            "config": config_snapshot,
            "derived": {
                "equilibration_steps": int(equilibration_steps),
                "exchange_frequency": int(exchange_frequency),
                "dcd_stride": int(dcd_stride),
                "save_state_frequency": int(save_state_frequency),
            },
            "resume_history": resume_history,
            "use_stub": bool(use_stub),
        }
        if resume_context_norm:
            resume_history.append(resume_context_norm)
            plan_payload["resume_context"] = resume_context_norm

        save_run_plan(run_dir, plan_payload)

        try:
            if use_stub:
                result, run_metadata = self._run_quick_sampling_stub(
                    run_label,
                    run_dir,
                    config,
                    restart_paths=restart_paths,
                    restart_temperature=restart_target_temperature,
                )
            else:
                result, run_metadata = self._run_real_sampling(
                    config=config,
                    run_label=run_label,
                    run_dir=run_dir,
                    save_state_frequency=int(save_state_frequency),
                    restart_paths=restart_paths,
                    restart_target_temperature=restart_target_temperature,
                    cv_kwargs=cv_kwargs,
                    dcd_stride=int(dcd_stride),
                )
        except Exception as exc:
            plan_payload["status"] = "failed"
            plan_payload["last_error"] = str(exc)
            save_run_plan(run_dir, plan_payload)
            raise
        else:
            plan_payload["status"] = "completed"
            plan_payload["last_completed_at"] = result.created_at
            save_run_plan(run_dir, plan_payload)

            if "stub_result" not in run_metadata:
                run_metadata["stub_result"] = bool(use_stub)
            if config.cv_model_bundle:
                run_metadata["cv_model_bundle"] = str(config.cv_model_bundle)
                run_metadata["cv_informed"] = True
            if resume_context_norm and "resumed_from" not in run_metadata:
                run_metadata["resumed_from"] = resume_context_norm.get(
                    "parent_run_id", resume_context_norm.get("run_id")
                )

            run_metadata["config_snapshot"] = config_snapshot
            run_metadata["run_plan"] = plan_payload
            self.state.upsert_run(run_metadata)
            return result

    def resume_run_from_checkpoint(self, run_id: str) -> SimulationResult:
        """Resume a corrupted run in-place using its latest checkpoint."""
        run_dir = (self.layout.sims_dir / run_id).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

        plan = load_run_plan(run_dir)
        if not plan:
            raise RuntimeError(
                f"No run plan metadata found for {run_id}. Cannot resume automatically."
            )
        if plan.get("use_stub"):
            raise RuntimeError("Stub runs cannot be resumed.")

        config_payload = plan.get("config") or plan.get("config_snapshot")
        if not config_payload:
            raise RuntimeError(
                f"Run {run_id} is missing configuration details required for resume."
            )

        config = SimulationConfig.from_snapshot(config_payload)
        checkpoint_path, checkpoint_step = self._latest_checkpoint_info(run_dir)
        derived = plan.get("derived") or {}
        exchange_frequency = int(
            derived.get("exchange_frequency")
            or config.exchange_frequency_steps
            or _derive_characteristics(config)[1]
        )
        resume_context = {
            "parent_run_id": run_id,
            "checkpoint_step": int(checkpoint_step),
            "checkpoint_md_steps": int(checkpoint_step) * int(exchange_frequency),
            "checkpoint_path": str(checkpoint_path),
            "requested_steps": int(config.steps),
            "resume_reason": "auto_recovery",
        }
        removed = self._prepare_run_directory_for_resume(run_dir)
        resume_context["deleted_files"] = removed

        config.start_from_checkpoint = checkpoint_path
        config.stub_result = False
        config.force_run_id = run_id
        config.label = run_id
        config.resume_context = resume_context
        config.save_state_frequency = int(
            derived.get("save_state_frequency") or _resolve_save_state_frequency(config)
        )

        logger.info(
            "Resuming run %s from checkpoint %s (step %d)",
            run_id,
            checkpoint_path.name,
            checkpoint_step,
        )

        return self.run_sampling(config)

    def _latest_checkpoint_info(self, run_dir: Path) -> Tuple[Path, int]:
        remd_dir = run_dir / "replica_exchange"
        if not remd_dir.exists():
            raise FileNotFoundError(f"replica_exchange directory missing for {run_dir}")
        checkpoints = sorted(remd_dir.glob("checkpoint_step_*.pkl"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {remd_dir}")
        latest = checkpoints[-1]
        try:
            step = int(latest.stem.split("_")[-1])
        except ValueError as exc:
            raise RuntimeError(f"Unrecognized checkpoint format: {latest.name}") from exc
        return latest.resolve(), step

    def _prepare_run_directory_for_resume(self, run_dir: Path) -> int:
        """Remove corrupted trajectory artifacts while keeping checkpoints."""
        remd_dir = run_dir / "replica_exchange"
        if not remd_dir.exists():
            return 0
        removed = 0
        patterns = ("replica_*.dcd", "demux_*.dcd")
        for pattern in patterns:
            for path in remd_dir.glob(pattern):
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    continue
        for artifact in (
            "analysis_results.json",
            "remd_diagnostics.json",
            "exchange_diagnostics.json",
        ):
            artifact_path = remd_dir / artifact
            if artifact_path.exists():
                artifact_path.unlink()
                removed += 1
        # Keep checkpoint files intact
        return removed

    def _run_quick_sampling_stub(
        self,
        run_label: str,
        run_dir: Path,
        config: SimulationConfig,
        *,
        restart_paths: Optional[Tuple[Path, Path]] = None,
        restart_temperature: Optional[float] = None,
    ) -> Tuple[SimulationResult, Dict[str, Any]]:
        """Generate a lightweight deterministic sampling result for quick-mode tests.

        Parameters
        ----------
        run_label:
            The label for this simulation run.
        run_dir:
            The simulation output directory.
        config:
            The simulation configuration.
        restart_paths:
            Optional tuple of (run_path, inputs_path) for restart snapshot.
        restart_temperature:
            Target temperature for restart snapshot extraction.

        Returns
        -------
        Tuple[SimulationResult, Dict[str, Any]]
            A tuple of (simulation_result, metadata_dict).
        """
        seed = (
            int(config.random_seed)
            if config.random_seed is not None
            else abs(hash(run_label)) % (2**32)
        )
        rng = np.random.default_rng(seed)
        template = md.load(str(config.pdb_path))
        base_coords = template.xyz[0]
        frames = max(5, min(50, int(config.steps) if config.steps else 5))

        rep_dir = (run_dir / "replica_exchange").resolve()
        ensure_directory(rep_dir)

        analysis_temperatures = [float(t) for t in config.temperatures]
        traj_files: List[Path] = []
        trajectories: List[md.Trajectory] = []
        for idx, temp in enumerate(config.temperatures):
            noise = 0.01 * rng.standard_normal((frames,) + base_coords.shape)
            coords = base_coords + noise
            traj = md.Trajectory(coords, template.topology)
            out_path = rep_dir / f"traj_{idx:02d}.dcd"
            traj.save_dcd(str(out_path))
            traj_files.append(out_path.resolve())
            trajectories.append(traj)

        restart_pdb_path: Optional[Path] = None
        restart_inputs_entry: Optional[Path] = None
        if restart_paths:
            run_path, inputs_path = restart_paths
            target_temp = (
                float(restart_temperature)
                if restart_temperature is not None
                else analysis_temperatures[0]
            )
            if not analysis_temperatures:
                raise ValueError("Cannot generate restart snapshot without temperatures.")
            target_idx = min(
                range(len(analysis_temperatures)),
                key=lambda i: abs(analysis_temperatures[i] - target_temp),
            )
            final_frame = trajectories[target_idx][-1]
            final_frame.save_pdb(str(run_path))
            shutil.copy2(run_path, inputs_path)
            restart_pdb_path = run_path.resolve()
            restart_inputs_entry = inputs_path.resolve()

        # Minimal diagnostics payload mirroring the real runner structure
        diag_payload = {
            "temperatures": analysis_temperatures,
            "exchange_attempts": int(max(1, frames - 1)),
            "exchange_accepted": int(max(0, frames // 2)),
            "per_pair_acceptance": [0.5 for _ in analysis_temperatures],
            "acceptance_mean": 0.5,
            "mean_abs_disp_per_10k_steps": 0.0,
            "mean_abs_disp_per_sweep": 0.0,
            "sparkline": [0.0 for _ in analysis_temperatures],
        }
        (rep_dir / "exchange_diagnostics.json").write_text(
            json.dumps(diag_payload, indent=2), encoding="utf-8"
        )

        created = _timestamp()
        result = SimulationResult(
            run_id=run_label,
            run_dir=run_dir.resolve(),
            pdb_path=Path(config.pdb_path).resolve(),
            traj_files=traj_files,
            analysis_temperatures=analysis_temperatures,
            steps=int(config.steps),
            created_at=created,
            restart_pdb_path=restart_pdb_path,
            restart_inputs_entry=restart_inputs_entry,
        )
        metadata = {
            "run_id": run_label,
            "run_dir": str(result.run_dir),
            "pdb": str(result.pdb_path),
            "temperatures": analysis_temperatures,
            "analysis_temperatures": analysis_temperatures,
            "steps": int(config.steps),
            "quick": True,
            "random_seed": (
                int(config.random_seed) if config.random_seed is not None else None
            ),
            "traj_files": [str(p) for p in traj_files],
            "created_at": created,
            "stub_result": True,
        }
        if restart_pdb_path:
            metadata["restart_pdb"] = str(restart_pdb_path)
        if restart_inputs_entry:
            metadata["restart_input_entry"] = str(restart_inputs_entry)
        if restart_temperature is not None:
            metadata["restart_temperature"] = float(restart_temperature)
        return result, metadata

    def _run_real_sampling(
        self,
        *,
        config: SimulationConfig,
        run_label: str,
        run_dir: Path,
        save_state_frequency: int,
        restart_paths: Optional[Tuple[Path, Path]],
        restart_target_temperature: Optional[float],
        cv_kwargs: Dict[str, Any],
        dcd_stride: int,
    ) -> Tuple[SimulationResult, Dict[str, Any]]:
        """Execute either REMD or single-temperature MD based on config."""

        recorded_temperatures: List[float] = [float(t) for t in config.temperatures]
        restart_pdb_path: Optional[Path] = None
        restart_inputs_entry: Optional[Path] = None

        if config.single_temperature_mode:
            from .utils import run_single_temperature_md

            target_temp = _resolve_primary_temperature(config)
            logger.info(
                "[sampling] Running single-temperature MD at %.1fK (no replicas)",
                target_temp,
            )
            traj_files, temp = run_single_temperature_md(
                pdb_file=str(config.pdb_path),
                output_dir=str(run_dir),
                temperature=target_temp,
                total_steps=int(config.steps),
                quick=bool(config.quick),
                random_seed=(
                    int(config.random_seed) if config.random_seed is not None else None
                ),
                jitter_start=bool(config.jitter_start),
                jitter_sigma_A=float(config.jitter_sigma_A),
                start_from_checkpoint=(
                    str(config.start_from_checkpoint)
                    if config.start_from_checkpoint
                    else None
                ),
                start_from_pdb=(
                    str(config.start_from_pdb) if config.start_from_pdb else None
                ),
                save_state_frequency=int(save_state_frequency),
                dcd_stride=int(dcd_stride),
                save_final_pdb=bool(config.save_restart_pdb),
                final_pdb_path=str(restart_paths[0]) if restart_paths else None,
                final_pdb_temperature=(
                    float(restart_target_temperature)
                    if restart_target_temperature is not None
                    else float(target_temp)
                ),
            )
            temps = [float(temp)]
            recorded_temperatures = [float(target_temp)]
        else:
            exchange_override = (
                int(config.exchange_frequency_steps)
                if config.exchange_frequency_steps is not None
                else None
            )
            traj_files, temps = run_replica_exchange(
                pdb_file=str(config.pdb_path),
                output_dir=str(run_dir),
                temperatures=[float(t) for t in config.temperatures],
                total_steps=int(config.steps),
                quick=bool(config.quick),
                random_seed=(
                    int(config.random_seed) if config.random_seed is not None else None
                ),
                jitter_start=bool(config.jitter_start),
                jitter_sigma_A=float(config.jitter_sigma_A),
                exchange_frequency_steps=exchange_override,
                temperature_schedule_mode=config.temperature_schedule_mode,
                start_from_checkpoint=(
                    str(config.start_from_checkpoint)
                    if config.start_from_checkpoint
                    else None
                ),
                start_from_pdb=(
                    str(config.start_from_pdb) if config.start_from_pdb else None
                ),
                save_final_pdb=bool(config.save_restart_pdb),
                final_pdb_path=str(restart_paths[0]) if restart_paths else None,
                final_pdb_temperature=restart_target_temperature,
                save_state_frequency=int(save_state_frequency),
                **cv_kwargs,
            )

        completed_at = _timestamp()

        if restart_paths:
            restart_pdb_path = restart_paths[0].resolve()
            if not restart_pdb_path.exists():
                raise FileNotFoundError(
                    f"Expected restart snapshot at {restart_pdb_path} was not produced."
                )
            target_copy = restart_paths[1]
            ensure_directory(target_copy.parent)
            shutil.copy2(restart_pdb_path, target_copy)
            restart_inputs_entry = target_copy.resolve()

        result = SimulationResult(
            run_id=run_label,
            run_dir=run_dir.resolve(),
            pdb_path=Path(config.pdb_path).resolve(),
            traj_files=_coerce_path_list(traj_files),
            analysis_temperatures=[float(t) for t in temps],
            steps=int(config.steps),
            created_at=completed_at,
            restart_pdb_path=restart_pdb_path,
            restart_inputs_entry=restart_inputs_entry,
        )
        run_metadata: Dict[str, Any] = {
            "run_id": run_label,
            "run_dir": str(result.run_dir),
            "pdb": str(result.pdb_path),
            "temperatures": recorded_temperatures,
            "analysis_temperatures": result.analysis_temperatures,
            "steps": int(config.steps),
            "quick": bool(config.quick),
            "random_seed": (
                int(config.random_seed) if config.random_seed is not None else None
            ),
            "traj_files": [str(p) for p in result.traj_files],
            "created_at": completed_at,
            "stub_result": False,
        }
        if restart_pdb_path:
            run_metadata["restart_pdb"] = str(restart_pdb_path)
        if restart_inputs_entry:
            run_metadata["restart_input_entry"] = str(restart_inputs_entry)
        if restart_target_temperature is not None:
            run_metadata["restart_temperature"] = float(restart_target_temperature)
        return result, run_metadata
