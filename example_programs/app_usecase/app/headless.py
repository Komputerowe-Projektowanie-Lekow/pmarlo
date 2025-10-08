from __future__ import annotations

"""Headless execution helpers mirroring the Streamlit analysis workflow."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .backend import BuildConfig, WorkflowBackend, WorkspaceLayout


def _make_layout(app_path: str | Path | None, workspace: str | Path | None) -> WorkspaceLayout:
    base = WorkspaceLayout.from_app_package(Path(app_path) if app_path else None)
    if workspace is None:
        return base
    ws = Path(workspace).resolve()
    layout = WorkspaceLayout(
        app_root=base.app_root,
        inputs_dir=base.inputs_dir,
        workspace_dir=ws,
        sims_dir=ws / "sims",
        shards_dir=ws / "shards",
        models_dir=ws / "models",
        bundles_dir=ws / "bundles",
        logs_dir=ws / "logs",
        state_path=ws / "state.json",
    )
    layout.ensure()
    return layout


def _build_backend(args: argparse.Namespace) -> WorkflowBackend:
    layout = _make_layout(args.app_root, args.workspace)
    return WorkflowBackend(layout)


def _parse_bins(entries: Sequence[str]) -> Dict[str, int]:
    bins: Dict[str, int] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Invalid bin specification '{item}', expected cv=count.")
        key, value = item.split("=", 1)
        bins[key.strip()] = int(value.strip())
    if not bins:
        bins = {"Rg": 72, "RMSD_ref": 72}
    return bins


def _load_optional_json(path: str | Path | None) -> Dict[str, object] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {file_path}, got {type(data).__name__}")
    return data


def _resolve_shards(
    backend: WorkflowBackend,
    shards: Sequence[str] | None,
    shard_group: int | None,
) -> List[Path]:
    if shards:
        return [Path(p).resolve() for p in shards]
    if shard_group is not None:
        if not (0 <= shard_group < len(backend.state.shards)):
            raise IndexError(f"Shard group index {shard_group} out of range.")
        entry = backend.state.shards[shard_group]
        paths = entry.get("paths", [])
        if not paths:
            raise ValueError(f"No shard paths recorded for group {shard_group}.")
        return [Path(p).resolve() for p in paths]
    raise ValueError("Provide explicit --shard paths or --shard-group index.")


def cmd_list_shards(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    if not backend.state.shards:
        print("No shard groups recorded.", file=sys.stdout)
        return 0
    for idx, entry in enumerate(backend.state.shards):
        paths: Iterable[str] = entry.get("paths", [])
        created_at = entry.get("created_at", "<unknown>")
        label = entry.get("label") or entry.get("group_name") or f"group-{idx}"
        print(f"[{idx}] {label} ({created_at}) - {len(list(paths))} shards", file=sys.stdout)
    return 0


def cmd_run_analysis(args: argparse.Namespace) -> int:
    backend = _build_backend(args)
    shard_paths = _resolve_shards(backend, args.shard, args.shard_group)
    bins = _parse_bins(args.bin or [])
    deeptica_params = _load_optional_json(args.deeptica_params)
    notes = _load_optional_json(args.notes)

    config = BuildConfig(
        lag=int(args.lag),
        bins=bins,
        seed=int(args.seed),
        temperature=float(args.temperature),
        learn_cv=bool(args.learn_cv),
        deeptica_params=deeptica_params,
        notes=notes or {},
        apply_cv_whitening=not args.no_whitening,
        cluster_mode=str(args.cluster_mode),
        n_microstates=int(args.microstates),
        reweight_mode=str(args.reweight_mode),
        fes_method=str(args.fes_method),
        fes_bandwidth=args.fes_bandwidth,
        fes_min_count_per_bin=int(args.fes_min_count),
    )

    artifact = backend.build_analysis(shard_paths, config)
    print(f"Bundle path: {artifact.bundle_path}", file=sys.stdout)
    if artifact.debug_dir:
        print(f"Debug artifacts: {artifact.debug_dir}", file=sys.stdout)
    if artifact.tau_frames is not None:
        print(f"tau (frames): {artifact.tau_frames}", file=sys.stdout)
    if (
        artifact.effective_tau_frames is not None
        and artifact.effective_tau_frames != artifact.tau_frames
    ):
        print(
            f"Effective tau (stride-adjusted): {artifact.effective_tau_frames}",
            file=sys.stdout,
        )
    if artifact.effective_stride_max is not None:
        print(
            f"Effective stride (max): {artifact.effective_stride_max}",
            file=sys.stdout,
        )
    if artifact.discretizer_fingerprint:
        print(
            "Discretizer fingerprint:",
            json.dumps(artifact.discretizer_fingerprint, indent=2),
            file=sys.stdout,
        )
    summary = artifact.debug_summary or {}
    stride_map = summary.get("effective_stride_map") or {}
    if stride_map:
        print("Effective stride map:", json.dumps(stride_map, indent=2), file=sys.stdout)
    preview = summary.get("preview_truncated") or []
    if preview:
        print(f"Preview-truncated shards: {preview}", file=sys.stdout)
    first_ts = summary.get("first_timestamps") or []
    last_ts = summary.get("last_timestamps") or []
    if first_ts or last_ts:
        first_repr = ', '.join(str(val) for val in first_ts) if first_ts else 'n/a'
        last_repr = ', '.join(str(val) for val in last_ts) if last_ts else 'n/a'
        print('First timestamps:', first_repr, '| Last timestamps:', last_repr, file=sys.stdout)
    if artifact.debug_summary:
        print("Debug summary:", file=sys.stdout)
        print(json.dumps(artifact.debug_summary, indent=2), file=sys.stdout)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless interface for the PMARLO app analysis workflow.",
    )
    parser.add_argument(
        "--app-root",
        type=Path,
        default=None,
        help="Path to the app/ directory (defaults to this file's package).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Override workspace directory (default: app_output under app root).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "list-shards",
        help="List shard groups recorded in the Streamlit state file.",
    )

    parser_run = subparsers.add_parser(
        "run-analysis",
        help="Run the analysis pipeline on selected shards.",
    )
    parser_run.add_argument(
        "--shard",
        action="append",
        help="Path to a shard JSON file (can be provided multiple times).",
    )
    parser_run.add_argument(
        "--shard-group",
        type=int,
        default=None,
        help="Index of a previously recorded shard group (from list-shards).",
    )
    parser_run.add_argument("--lag", type=int, default=10, help="Lag time (frames).")
    parser_run.add_argument("--seed", type=int, default=2025, help="Build seed.")
    parser_run.add_argument(
        "--temperature", type=float, default=300.0, help="Analysis temperature (K)."
    )
    parser_run.add_argument(
        "--bin",
        action="append",
        metavar="CV=COUNT",
        help="Bin specification, e.g. --bin Rg=72 --bin RMSD_ref=72.",
    )
    parser_run.add_argument(
        "--learn-cv",
        action="store_true",
        help="Enable Deeptica CV learning as part of the build.",
    )
    parser_run.add_argument(
        "--deeptica-params",
        type=Path,
        default=None,
        help="JSON file with Deeptica configuration overrides.",
    )
    parser_run.add_argument(
        "--notes",
        type=Path,
        default=None,
        help="JSON file of arbitrary notes to attach to the build.",
    )
    parser_run.add_argument(
        "--no-whitening",
        action="store_true",
        help="Disable enforced CV whitening during analysis.",
    )
    parser_run.add_argument(
        "--cluster-mode",
        type=str,
        default="kmeans",
        help="Clustering mode for MSM (default: kmeans).",
    )
    parser_run.add_argument(
        "--microstates",
        type=int,
        default=150,
        help="Number of microstates for MSM construction.",
    )
    parser_run.add_argument(
        "--reweight-mode",
        type=str,
        default="MBAR",
        help="Reweighting strategy (MBAR or TRAM).",
    )
    parser_run.add_argument(
        "--fes-method",
        type=str,
        default="kde",
        help="Free energy surface method (kde or histogram).",
    )
    parser_run.add_argument(
        "--fes-bandwidth",
        default="scott",
        help="Bandwidth parameter for the FES estimator.",
    )
    parser_run.add_argument(
        "--fes-min-count",
        type=int,
        default=1,
        help="Minimum count per FES bin before smoothing.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list-shards":
        return cmd_list_shards(args)
    if args.command == "run-analysis":
        return cmd_run_analysis(args)
    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
