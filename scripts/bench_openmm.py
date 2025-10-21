#!/usr/bin/env python
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np
import os
import torch
from openmm import Platform, unit
from openmm.app import ForceField, PDBFile, Simulation
from openmm import VerletIntegrator
import yaml

from pmarlo.replica_exchange.system_builder import create_system
from pmarlo.settings import load_defaults, resolve_feature_spec_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OpenMM with optional CV bias")
    parser.add_argument("--platform", default="CPU", help="OpenMM platform (default: CPU)")
    parser.add_argument("--with-bias", choices=["yes", "no"], default="yes", help="Enable CV bias (default: yes)")
    parser.add_argument("--steps", type=int, default=5000, help="Number of MD steps (default: 5000)")
    parser.add_argument("--torch-threads", type=int, default=None, help="Override Torch threads")
    parser.add_argument("--model", type=Path, default=None, help="Path to TorchScript CV model")
    parser.add_argument("--report-interval", type=int, default=1000, help="Logging interval (default: 1000 steps)")
    return parser.parse_args()


def write_config(spec_path: Path, enable_bias: bool, torch_threads: int | None, tmpdir: Path) -> Path:
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


def load_system(pdb_path: Path, model_path: Path | None, platform_name: str) -> Simulation:
    pdb = PDBFile(str(pdb_path))
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = create_system(pdb, forcefield, cv_model_path=str(model_path) if model_path else None)
    integrator = VerletIntegrator(2.0 * unit.femtoseconds)
    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
    return simulation


def gather_stats(simulation: Simulation, steps: int, report_interval: int, model_path: Path | None):
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
            state = simulation.context.getState(getPositions=True, getEnergy=True, groups={1})
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            bias_samples.append(float(energy))
            positions = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
            pos_tensor = torch.tensor(positions, dtype=torch.float32)
            box = torch.eye(3, dtype=torch.float32)
            with torch.inference_mode():
                cvs = module.compute_cvs(pos_tensor, box)
            cv_samples.append(cvs.detach().cpu().numpy())
    elapsed = time.perf_counter() - start
    steps_per_second = steps / elapsed if elapsed > 0 else float("nan")
    return bias_samples, cv_samples, steps_per_second


def summarise(bias_samples, cv_samples, steps_per_second, config, bias_enabled: bool, platform: str, steps: int):
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

    print("Benchmark summary")
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


if __name__ == "__main__":
    args = parse_args()
    bias_enabled = args.with_bias == "yes"
    if bias_enabled and args.model is None:
        raise SystemExit("--model is required when --with-bias=yes")

    base_spec = resolve_feature_spec_path()
    pdb_path = Path(__file__).resolve().parents[1] / "tests" / "_assets" / "3gd8-fixed.pdb"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cfg_path = write_config(base_spec, bias_enabled, args.torch_threads, tmp_path)
        os.environ["PMARLO_CONFIG_FILE"] = str(cfg_path)
        load_defaults.cache_clear()

        simulation = load_system(pdb_path, args.model if bias_enabled else None, args.platform)
        bias_samples, cv_samples, steps_per_second = gather_stats(
            simulation,
            steps=args.steps,
            report_interval=max(1, args.report_interval),
            model_path=args.model if bias_enabled else None,
        )
        config = load_defaults()
        summarise(bias_samples, cv_samples, steps_per_second, config, bias_enabled, args.platform, args.steps)
