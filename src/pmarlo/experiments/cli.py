import argparse
import json
import logging
from pathlib import Path

from .msm import MSMConfig, run_msm_experiment
from .replica_exchange import ReplicaExchangeConfig, run_replica_exchange_experiment
from .simulation import SimulationConfig, run_simulation_experiment
from .utils import tests_data_dir

# CLI sets logging level; modules themselves do not configure basicConfig


def _tests_data_dir() -> Path:
    """Return the path to ``tests/data`` for use as CLI defaults."""

    return tests_data_dir()


def main():
    parser = argparse.ArgumentParser(
        description="PMARLO Experiments Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v: INFO, -vv: DEBUG)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1) Simulation experiment
    sim = sub.add_parser("simulation", help="Run single-T simulation experiment")
    sim.add_argument("--pdb", default=str(_tests_data_dir() / "3gd8-fixed.pdb"))
    sim.add_argument("--steps", type=int, default=500)
    sim.add_argument("--out", default="experiments_output/simulation")
    sim.add_argument("--no-meta", action="store_true", help="Disable metadynamics")

    # 2) Replica exchange experiment
    remd = sub.add_parser("remd", help="Run replica exchange experiment")
    remd.add_argument("--pdb", default=str(_tests_data_dir() / "3gd8-fixed.pdb"))
    remd.add_argument("--steps", type=int, default=800)
    remd.add_argument("--equil", type=int, default=200)
    remd.add_argument("--freq", type=int, default=50, help="Exchange frequency")
    remd.add_argument("--out", default="experiments_output/replica_exchange")
    remd.add_argument("--no-meta", action="store_true", help="Disable metadynamics")
    remd.add_argument(
        "--tmin",
        type=float,
        default=300.0,
        help="Minimum temperature for ladder if --temps not given",
    )
    remd.add_argument(
        "--tmax",
        type=float,
        default=350.0,
        help="Maximum temperature for ladder if --temps not given",
    )
    remd.add_argument(
        "--nrep",
        type=int,
        default=6,
        help="Number of replicas for ladder if --temps not given",
    )

    # 3) MSM experiment
    msm = sub.add_parser("msm", help="Run MSM experiment on trajectories")
    msm.add_argument(
        "--traj",
        nargs="+",
        default=[str(_tests_data_dir() / "traj.dcd")],
        help="Trajectory files (DCD)",
    )
    msm.add_argument("--top", default=str(_tests_data_dir() / "3gd8-fixed.pdb"))
    msm.add_argument("--clusters", type=int, default=60)
    msm.add_argument("--lag", type=int, default=20)
    msm.add_argument("--out", default="experiments_output/msm")
    msm.add_argument("--stride", type=int, default=1, help="Trajectory frame stride")
    msm.add_argument(
        "--atom-selection",
        default=None,
        help="MDTraj atom selection string to subset atoms",
    )

    args = parser.parse_args()

    # Configure logging early
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if args.cmd == "simulation":
        cfg = SimulationConfig(
            pdb_file=args.pdb,
            output_dir=args.out,
            steps=args.steps,
            use_metadynamics=not args.no_meta,
        )
        result = run_simulation_experiment(cfg)
    elif args.cmd == "remd":
        cfg = ReplicaExchangeConfig(
            pdb_file=args.pdb,
            output_dir=args.out,
            total_steps=args.steps,
            equilibration_steps=args.equil,
            exchange_frequency=args.freq,
            use_metadynamics=not args.no_meta,
            tmin=args.tmin,
            tmax=args.tmax,
            nreplicas=args.nrep,
        )
        result = run_replica_exchange_experiment(cfg)
    else:
        cfg = MSMConfig(
            trajectory_files=args.traj,
            topology_file=args.top,
            output_dir=args.out,
            n_clusters=args.clusters,
            lag_time=args.lag,
            stride=args.stride,
            atom_selection=args.atom_selection,
        )
        result = run_msm_experiment(cfg)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
