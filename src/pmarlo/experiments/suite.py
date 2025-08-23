from __future__ import annotations

"""
Predefined benchmark suite for PMARLO experiments.

This module defines a fixed set of benchmark cases (no runtime parameter picking),
so Kubernetes Indexed Jobs can map index -> case deterministically and write
results into the standard algorithm output roots under `experiments_output/`.

Usage:
  python -m pmarlo.experiments.suite --index 0

Indexes 0..5 map to 2 variants for each of 3 algorithms:
  0: simulation-A
  1: simulation-B
  2: remd-A
  3: remd-B
  4: msm-A
  5: msm-B
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from .msm import MSMConfig, run_msm_experiment
from .replica_exchange import ReplicaExchangeConfig, run_replica_exchange_experiment
from .simulation import SimulationConfig, run_simulation_experiment
from .utils import tests_data_dir

AlgorithmName = Literal["simulation", "remd", "msm"]


class SuiteResult(TypedDict):
    case_id: str
    algorithm: AlgorithmName
    result: dict


@dataclass(frozen=True)
class SuiteCase:
    case_id: str
    algorithm: AlgorithmName
    # Parameters specific to each algorithm; we keep a superset for clarity.
    # Simulation
    sim_steps: int | None = None
    sim_use_metadynamics: bool | None = None
    # REMD
    remd_steps: int | None = None
    remd_equil: int | None = None
    remd_nrep: int | None = None
    remd_tmin: float | None = None
    remd_tmax: float | None = None
    # MSM
    msm_clusters: int | None = None
    msm_lag: int | None = None


def _tests_pdb() -> str:
    """Return the path to the test PDB file."""

    return str(tests_data_dir() / "3gd8-fixed.pdb")


def _tests_traj() -> list[str]:
    """Return paths to the default test trajectories."""

    return [str(tests_data_dir() / "traj.dcd")]


def get_suite_cases() -> list[SuiteCase]:
    return [
        # simulation A
        SuiteCase(
            case_id="simulation-A",
            algorithm="simulation",
            sim_steps=500,
            sim_use_metadynamics=True,
        ),
        # simulation B
        SuiteCase(
            case_id="simulation-B",
            algorithm="simulation",
            sim_steps=800,
            sim_use_metadynamics=False,
        ),
        # remd A
        SuiteCase(
            case_id="remd-A",
            algorithm="remd",
            remd_steps=800,
            remd_equil=200,
            remd_nrep=8,
            remd_tmin=300.0,
            remd_tmax=350.0,
        ),
        # remd B
        SuiteCase(
            case_id="remd-B",
            algorithm="remd",
            remd_steps=800,
            remd_equil=200,
            remd_nrep=10,
            remd_tmin=300.0,
            remd_tmax=340.0,
        ),
        # msm A
        SuiteCase(
            case_id="msm-A",
            algorithm="msm",
            msm_clusters=60,
            msm_lag=20,
        ),
        # msm B
        SuiteCase(
            case_id="msm-B",
            algorithm="msm",
            msm_clusters=80,
            msm_lag=30,
        ),
    ]


def run_suite_case(index: int) -> SuiteResult:
    cases = get_suite_cases()
    if index < 0 or index >= len(cases):
        raise IndexError(f"Suite index {index} out of range [0, {len(cases) - 1}]")
    c = cases[index]

    if c.algorithm == "simulation":
        sim_cfg = SimulationConfig(
            pdb_file=_tests_pdb(),
            # Keep algorithm root stable; per-run timestamped subdir is created
            # internally by the experiment function
            steps=int(c.sim_steps or 500),
            use_metadynamics=bool(c.sim_use_metadynamics is True),
            seed=0,
        )
        res = run_simulation_experiment(sim_cfg)
    elif c.algorithm == "remd":
        remd_cfg = ReplicaExchangeConfig(
            pdb_file=_tests_pdb(),
            total_steps=int(c.remd_steps or 800),
            equilibration_steps=int(c.remd_equil or 200),
            nreplicas=int(c.remd_nrep or 6),
            tmin=float(c.remd_tmin or 300.0),
            tmax=float(c.remd_tmax or 350.0),
            seed=0,
        )
        res = run_replica_exchange_experiment(remd_cfg)
    else:
        msm_cfg = MSMConfig(
            trajectory_files=_tests_traj(),
            topology_file=_tests_pdb(),
            n_clusters=int(c.msm_clusters or 60),
            lag_time=int(c.msm_lag or 20),
            seed=0,
        )
        res = run_msm_experiment(msm_cfg)

    return {"case_id": c.case_id, "algorithm": c.algorithm, "result": res}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a predefined PMARLO benchmark suite case by index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    out = run_suite_case(args.index)
    print(json.dumps(out, indent=2))

    # Write a small registry entry to help discover artifacts post-run
    try:
        algo = out.get("algorithm") if isinstance(out, dict) else None
        case_id = out.get("case_id") if isinstance(out, dict) else None
        result = out.get("result") if isinstance(out, dict) else {}
        run_dir = None
        if isinstance(result, dict):
            # Different experiments return run_dir or similar
            run_dir = result.get("run_dir") or result.get("trajectories_dir")
        idx_env = os.getenv("JOB_INDEX") or os.getenv("JOB_COMPLETION_INDEX")
        pod_name = os.getenv("HOSTNAME")
        registry_root = Path("experiments_output") / "_registry"
        registry_root.mkdir(parents=True, exist_ok=True)
        fname_parts = [str(idx_env or args.index)]
        if case_id:
            fname_parts.append(str(case_id))
        if algo:
            fname_parts.append(str(algo))
        registry_path = registry_root / ("-".join(fname_parts) + ".json")
        payload = {
            "job_index": idx_env if idx_env is not None else args.index,
            "algorithm": algo,
            "case_id": case_id,
            "run_dir": run_dir,
            "pod_name": pod_name,
        }
        with open(registry_path, "w", encoding="utf-8") as rf:
            json.dump(payload, rf, indent=2)
        # Also print a single-line locator for easy grep in logs
        locator = {
            "ARTIFACT_DIR": run_dir,
            "ALGO": algo,
            "CASE": case_id,
            "JOB_INDEX": idx_env if idx_env is not None else args.index,
            "POD": pod_name,
        }
        print("PMARLO_ARTIFACT_LOCATOR " + json.dumps(locator, separators=(",", ":")))
    except Exception:
        # Non-fatal convenience feature
        pass


if __name__ == "__main__":
    main()
