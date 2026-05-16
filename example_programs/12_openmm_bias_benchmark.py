#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from _example_support import assets_path, ensure_src_on_path, example_output_dir
from openmm import Platform, VerletIntegrator, unit
from openmm.app import ForceField, PDBFile, Simulation

ensure_src_on_path()

from pmarlo.features.deeptica.export import load_cv_model_info
from pmarlo.replica_exchange.system_builder import (
    create_system,
    resolve_cv_model_torchscript_path,
)
from pmarlo.settings import load_defaults, resolve_feature_spec_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenMM with optional CV bias"
    )
    parser.add_argument(
        "--platform", default="CPU", help="OpenMM platform (default: CPU)"
    )
    parser.add_argument(
        "--with-bias",
        choices=["yes", "no"],
        default="yes",
        help="Enable CV bias (default: yes)",
    )
    parser.add_argument(
        "--steps", type=int, default=5000, help="Number of MD steps (default: 5000)"
    )
    parser.add_argument(
        "--torch-threads", type=int, default=None, help="Override Torch threads"
    )
    parser.add_argument(
        "--model", type=Path, default=None, help="Path to TorchScript CV model"
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=1000,
        help="Logging interval (default: 1000 steps)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for benchmark summary JSON",
    )
    parser.add_argument(
        "--pdb",
        type=Path,
        default=None,
        help="Path to the solvated PDB structure compatible with the CV model (default: bundled 3gd8-fixed.pdb)",
    )
    return parser.parse_args()


def validate_model_bundle(model_path: Path) -> None:
    """Validate that the TorchScript model exists and report any companion files."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    stem = model_path.stem
    parent = model_path.parent

    scaler_pt = parent / f"{stem}.scaler.pt"
    scaler_npz = parent / f"{stem}_scaler.npz"
    json_file = parent / f"{stem}.json"

    available = []
    if scaler_pt.exists():
        available.append(f"Scaler (TorchScript) : {scaler_pt}")
    elif scaler_npz.exists():
        available.append(f"Scaler (NumPy)       : {scaler_npz}")

    if json_file.exists():
        available.append(f"Metadata            : {json_file}")

    print(f"Model bundle inspection:")
    print(f"  Model: {model_path}")
    if available:
        for line in available:
            print(f"  {line}")
    else:
        print(
            "  No scaler/metadata files detected; continuing with TorchScript-only bundle."
        )


def write_config(
    spec_path: Path, enable_bias: bool, torch_threads: int | None, tmpdir: Path
) -> Path:
    defaults = load_defaults()
    config = dict(defaults)
    config.update(
        {
            "enable_cv_bias": bool(enable_bias),
            "feature_spec_path": str(spec_path),
        }
    )
    if torch_threads is not None:
        if torch_threads <= 0:
            raise ValueError("torch_threads must be positive")
        config["torch_threads"] = int(torch_threads)
    cfg_path = tmpdir / "pmarlo_bench_config.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return cfg_path


def load_system(
    pdb_path: Path, model_path: Path | None, platform_name: str
) -> Simulation:
    pdb = PDBFile(str(pdb_path))
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = create_system(
        pdb, forcefield, cv_model_path=str(model_path) if model_path else None
    )
    integrator = VerletIntegrator(2.0 * unit.femtoseconds)
    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy(maxIterations=200)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    return simulation


def gather_stats(
    simulation: Simulation, steps: int, report_interval: int, model_path: Path | None
):
    bias_samples: list[float] = []
    cv_samples: list[np.ndarray] = []
    remaining = steps
    module = torch.jit.load(str(model_path), map_location="cpu") if model_path else None
    if module is not None:
        module.eval()
    start = time.perf_counter()
    while remaining > 0:
        chunk = min(report_interval, remaining)
        simulation.step(chunk)
        remaining -= chunk
        if module is not None:
            state = simulation.context.getState(
                getPositions=True, getEnergy=True, groups={1}
            )
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            bias_samples.append(float(energy))
            positions = np.array(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            )
            pos_tensor = torch.tensor(positions, dtype=torch.float32)
            box = torch.eye(3, dtype=torch.float32)
            with torch.inference_mode():
                cvs = module.compute_cvs(pos_tensor, box)
            cv_samples.append(cvs.detach().cpu().numpy())
    elapsed = time.perf_counter() - start
    steps_per_second = steps / elapsed if elapsed > 0 else float("nan")
    return bias_samples, cv_samples, steps_per_second


def summarise(
    bias_samples,
    cv_samples,
    steps_per_second,
    config,
    bias_enabled: bool,
    platform: str,
    steps: int,
):
    def mean_std(data):
        if not data:
            return float("nan"), float("nan")
        arr = np.array(data)
        return float(arr.mean()), float(arr.std())

    bias_mean, bias_std = mean_std(bias_samples)
    if cv_samples:
        stacked = np.concatenate(cv_samples, axis=0)
        cv_mean = stacked.mean(axis=0)
        cv_std = stacked.std(axis=0)
    else:
        cv_mean = cv_std = np.array([])

    print("\nBenchmark summary")
    print("------------------")
    print(f"Platform           : {platform}")
    print(f"Bias enabled       : {bias_enabled}")
    print(f"Steps              : {steps}")
    print(f"Torch threads      : {config['torch_threads']}")
    print(f"Precision          : {config['precision']}")
    print(f"Steps / second     : {steps_per_second:.2f}")
    if bias_enabled:
        print(f"Bias energy mean   : {bias_mean:.6f} kJ/mol")
        print(f"Bias energy std    : {bias_std:.6f} kJ/mol")
        print(f"CV mean            : {np.array2string(cv_mean, precision=4)}")
        print(f"CV std             : {np.array2string(cv_std, precision=4)}")
    return {
        "platform": platform,
        "bias_enabled": bool(bias_enabled),
        "steps": int(steps),
        "torch_threads": int(config["torch_threads"]),
        "precision": str(config["precision"]),
        "steps_per_second": float(steps_per_second),
        "bias_energy_mean": float(bias_mean),
        "bias_energy_std": float(bias_std),
        "cv_mean": cv_mean.tolist(),
        "cv_std": cv_std.tolist(),
    }


if __name__ == "__main__":
    args = parse_args()
    bias_enabled = args.with_bias == "yes"

    if bias_enabled and args.model is None:
        raise SystemExit("--model is required when --with-bias=yes")

    # Validate model bundle files exist
    if bias_enabled and args.model is not None:
        try:
            validate_model_bundle(args.model)
        except FileNotFoundError as e:
            raise SystemExit(f"Error: {e}")

    output_dir = args.output_dir or example_output_dir("12_openmm_bias_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    default_pdb = assets_path("3gd8-fixed.pdb")
    pdb_path = args.pdb if args.pdb is not None else default_pdb
    if not pdb_path.exists():
        raise SystemExit(f"PDB file not found: {pdb_path}")
    requested_model_path = args.model

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        spec_path = resolve_feature_spec_path()

        torchscript_model_path: Path | None = None
        if bias_enabled and requested_model_path is not None:
            torchscript_model_path = Path(
                resolve_cv_model_torchscript_path(requested_model_path)
            )
            bundle_info = load_cv_model_info(
                torchscript_model_path.parent, torchscript_model_path.stem
            )
            model_spec = bundle_info.get("metadata", {}).get("feature_spec")
            if model_spec:
                spec_override = tmp_path / "feature_spec.yaml"
                spec_override.write_text(yaml.safe_dump(model_spec), encoding="utf-8")
                spec_path = spec_override
            if torchscript_model_path.resolve() != Path(requested_model_path).resolve():
                print(f"Resolved TorchScript CV model: {torchscript_model_path}")

        cfg_path = write_config(spec_path, bias_enabled, args.torch_threads, tmp_path)
        os.environ["PMARLO_CONFIG_FILE"] = str(cfg_path)
        load_defaults.cache_clear()

        simulation = load_system(
            pdb_path,
            torchscript_model_path if bias_enabled else None,
            args.platform,
        )
        bias_samples, cv_samples, steps_per_second = gather_stats(
            simulation,
            steps=args.steps,
            report_interval=max(1, args.report_interval),
            model_path=torchscript_model_path if bias_enabled else None,
        )
        config = load_defaults()
        summary = summarise(
            bias_samples,
            cv_samples,
            steps_per_second,
            config,
            bias_enabled,
            args.platform,
            args.steps,
        )
        summary_path = output_dir / "benchmark_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary: {summary_path}")
