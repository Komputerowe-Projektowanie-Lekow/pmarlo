from pathlib import Path
from typing import Optional, Dict, Any
from .types import SimulationResult
from .state import PersistentState
from .layout import WorkspaceLayout


def load_run(self, run_id: str) -> Optional[SimulationResult]:
    """Best-effort reconstruction of a previous simulation result."""

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
        # Fallback: scan standard REMD output locations
        candidates: List[Path] = []
        replica_dir = run_dir / "replica_exchange"
        demux_dir = run_dir
        candidates.extend(sorted(replica_dir.rglob("*.dcd")))
        candidates.extend(sorted(replica_dir.rglob("*.nc")))
        candidates.extend(sorted(demux_dir.glob("demux_*.*")))
        traj_files = candidates
    if not traj_files:
        return None

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
    """Delete a simulation run and its associated files."""
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
) -> "SimulationResult":
    """Run a short simulation for testing purposes.

    Parameters
    ----------
    use_stub:
        When ``True`` (default if ``quick`` is ``True``), generate synthetic
        trajectories rather than invoking the full REMD stack. Setting this to
        ``False`` forces a real simulation even in quick mode.
    """
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

def _plan_restart_snapshot_paths(
        self,
        *,
        run_label: str,
        run_dir: Path,
        source_pdb: Path,
) -> Tuple[Path, Path]:
    """Determine output locations for restart snapshots."""
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
    base_label = _slugify(config.label) or f"run-{_timestamp()}"

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
                "⚠️  PyTorch is running on CPU only!\n"
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

        logger.info("✓ CV bias potential loaded successfully")
        logger.info(f"  Model path: {cv_info['model_path']}")
        logger.info(f"  CV dimensions: {cv_info['config']['cv_dim']}")
        logger.info(f"  Bias type: {cv_info['config'].get('bias_type', 'harmonic_expansion')}")
        logger.info(f"  Bias strength: {cv_info['config'].get('bias_strength', 10.0):.1f} kJ/mol")
        logger.info("\nBias physics:")
        logger.info("  E_bias = k * sum(cv_i^2)")
        logger.info("  Forces: F = -∇E_bias (computed by OpenMM)")
        logger.info("  Purpose: Repulsive bias → explore diverse conformations")
        logger.info("\n⚠️  IMPORTANT: The model expects MOLECULAR FEATURES as input")
        logger.info("  (distances, angles, dihedrals), not raw atomic positions.")
        logger.info("  Feature extraction must be configured in OpenMM system.")
    run_label = base_label
    if config.random_seed is not None:
        run_label = f"{base_label}-seed-{int(config.random_seed)}"
    elif use_stub:
        run_label = f"{base_label}-stub-{len(self.state.runs)}"
    run_dir = (self.layout.sims_dir / run_label).resolve()
    ensure_directory(run_dir)

    restart_paths: Optional[tuple[Path, Path]] = None
    restart_target_temperature: Optional[float] = None
    if config.save_restart_pdb:
        if not config.temperatures:
            raise ValueError("Temperature ladder required when saving restart PDB.")
        restart_target_temperature = (
            float(config.restart_temperature)
            if config.restart_temperature is not None
            else float(config.temperatures[0])
        )
        restart_paths = self._plan_restart_snapshot_paths(
            run_label=run_label,
            run_dir=run_dir,
            source_pdb=Path(config.pdb_path),
        )

    if use_stub:
        result, metadata = self._run_quick_sampling_stub(
            run_label,
            run_dir,
            config,
            restart_paths=restart_paths,
            restart_temperature=restart_target_temperature,
        )
        self.state.append_run(metadata)
        return result

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
        exchange_frequency_steps=(
            int(config.exchange_frequency_steps)
            if config.exchange_frequency_steps is not None
            else None
        ),
        temperature_schedule_mode=config.temperature_schedule_mode,
        start_from_pdb=(
            str(config.start_from_pdb) if config.start_from_pdb else None
        ),
        save_final_pdb=bool(config.save_restart_pdb),
        final_pdb_path=str(restart_paths[0]) if restart_paths else None,
        final_pdb_temperature=restart_target_temperature,
        **cv_kwargs,
    )
    created = _timestamp()
    restart_pdb_path: Optional[Path] = None
    restart_inputs_entry: Optional[Path] = None
    if restart_paths:
        restart_pdb_path = restart_paths[0].resolve()
        if not restart_pdb_path.exists():
            raise FileNotFoundError(
                f"Expected restart snapshot at {restart_pdb_path} was not produced."
            )
        target_copy = restart_paths[1]
        # Ensure parent exists (already ensured in planner but guard path operations)
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
        created_at=created,
        restart_pdb_path=restart_pdb_path,
        restart_inputs_entry=restart_inputs_entry,
    )
    run_metadata = {
        "run_id": run_label,
        "run_dir": str(result.run_dir),
        "pdb": str(result.pdb_path),
        "temperatures": [float(t) for t in config.temperatures],
        "analysis_temperatures": result.analysis_temperatures,
        "steps": int(config.steps),
        "quick": bool(config.quick),
        "random_seed": (
            int(config.random_seed) if config.random_seed is not None else None
        ),
        "traj_files": [str(p) for p in result.traj_files],
        "created_at": created,
        "stub_result": bool(use_stub),
    }
    if restart_pdb_path:
        run_metadata["restart_pdb"] = str(restart_pdb_path)
    if restart_inputs_entry:
        run_metadata["restart_input_entry"] = str(restart_inputs_entry)
    if restart_target_temperature is not None:
        run_metadata["restart_temperature"] = float(restart_target_temperature)

    # Add CV model reference if used
    if config.cv_model_bundle:
        run_metadata["cv_model_bundle"] = str(config.cv_model_bundle)
        run_metadata["cv_informed"] = True

    self.state.append_run(run_metadata)
    return result

def _run_quick_sampling_stub(
    self,
    run_label: str,
    run_dir: Path,
    config: SimulationConfig,
    *,
    restart_paths: Optional[Tuple[Path, Path]] = None,
    restart_temperature: Optional[float] = None,
) -> tuple[SimulationResult, Dict[str, Any]]:
    """Generate a lightweight deterministic sampling result for quick-mode tests."""

    import numpy as np
    import mdtraj as md

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
    traj_files: list[Path] = []
    trajectories: list[md.Trajectory] = []
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
    import json as _json

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
        _json.dumps(diag_payload, indent=2), encoding="utf-8"
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

class SimulationMixin:
    """Methods for running and managing MD simulations."""

    # Requires: self.state, self.layout

    def run_simulation(
            self,
            pdb_path: Path,
            steps: int,
            temperatures: list[float],
            ...
    ) -> SimulationResult:

    # ... your run_simulation implementation

    def load_run(self, run_id: str) -> Optional[SimulationResult]:

    # ... your load_run implementation

    def delete_simulation(self, index: int) -> bool:
# ... your delete_simulation implementation
