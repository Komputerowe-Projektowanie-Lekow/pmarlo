import argparse
import json
import logging
from pathlib import Path

from .utils import default_output_root, tests_data_dir

# CLI sets logging level; modules themselves do not configure basicConfig


def _tests_data_dir() -> Path:
    """Return the path to ``tests/_assets`` for use as CLI defaults."""

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
    sim.add_argument("--out", default=f"{default_output_root()}/simulation")
    sim.add_argument("--no-meta", action="store_true", help="Disable metadynamics")

    # 2) Replica exchange experiment
    remd = sub.add_parser("remd", help="Run replica exchange experiment")
    remd.add_argument("--pdb", default=str(_tests_data_dir() / "3gd8-fixed.pdb"))
    remd.add_argument("--steps", type=int, default=800)
    remd.add_argument("--equil", type=int, default=200)
    remd.add_argument("--freq", type=int, default=50, help="Exchange frequency")
    remd.add_argument("--out", default=f"{default_output_root()}/replica_exchange")
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
    msm.add_argument("--out", default=f"{default_output_root()}/msm")
    msm.add_argument("--stride", type=int, default=1, help="Trajectory frame stride")
    msm.add_argument(
        "--atom-selection",
        default=None,
        help="MDTraj atom selection string to subset atoms",
    )
    msm.add_argument(
        "--fes-smoothing-mode",
        choices=["never", "auto", "always"],
        default="never",
        help="FES smoothing policy. Default: never (no smoothing unless explicitly enabled).",
    )
    msm.add_argument(
        "--fes-target-sd-kT",
        type=float,
        default=None,
        help="Target SD[F] in kT units to trigger smoothing. If None, uses 0.5 when mode!=never.",
        dest="fes_target_sd_kT",
    )
    msm.add_argument(
        "--fes-alpha",
        type=float,
        default=1e-6,
        help="Dirichlet pseudocount per bin for uncertainty calculation.",
    )
    msm.add_argument(
        "--fes-h0",
        type=float,
        default=1.2,
        help="Base bandwidth (in grid-bin units).",
    )
    msm.add_argument(
        "--fes-ess-ref",
        type=float,
        default=50.0,
        help="Reference ESS for h0 scaling.",
    )
    msm.add_argument(
        "--fes-h-min",
        type=float,
        default=0.4,
        help="Minimum local bandwidth.",
    )
    msm.add_argument(
        "--fes-h-max",
        type=float,
        default=3.0,
        help="Maximum local bandwidth.",
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
        from .simulation import SimulationConfig, run_simulation_experiment

        cfg = SimulationConfig(
            pdb_file=args.pdb,
            output_dir=args.out,
            steps=args.steps,
            use_metadynamics=not args.no_meta,
        )
        result = run_simulation_experiment(cfg)
    elif args.cmd == "remd":
        from .replica_exchange import (
            ReplicaExchangeConfig,
            run_replica_exchange_experiment,
        )

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
        from .msm import MSMConfig, run_msm_experiment

        cfg = MSMConfig(
            trajectory_files=args.traj,
            topology_file=args.top,
            output_dir=args.out,
            n_clusters=args.clusters,
            lag_time=args.lag,
            stride=args.stride,
            atom_selection=args.atom_selection,
            fes_smoothing_mode=args.fes_smoothing_mode,
            fes_target_sd_kT=args.fes_target_sd_kT,
            fes_alpha=args.fes_alpha,
            fes_h0=args.fes_h0,
            fes_ess_ref=args.fes_ess_ref,
            fes_h_min=args.fes_h_min,
            fes_h_max=args.fes_h_max,
        )
        result = run_msm_experiment(cfg)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
