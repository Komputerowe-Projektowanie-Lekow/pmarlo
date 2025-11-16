"""Example program that mirrors the Streamlit conformations workflow.

This script selects *all* generated shards under
``pmarlo_webapp/app_output/shards`` (20 runs / 33 shards / 18,825 frames) and
executes the TPT-based conformations analysis with the DBSCAN configuration that
was previously driven via the UI:

    - Lag time: 110 steps
    - TICA components: 5
    - Temperature: 300 K
    - DBSCAN (min_samples=5, leaf_size=30, auto-tuned ``eps``)
    - N_metastable: 8
    - Max pathways: 8
    - Bootstrap samples: 50

Outputs are written to ``example_programs/programs_outputs/conformations`` so the
results can be inspected or versioned alongside other example artefacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from _example_support import ensure_src_on_path, project_root

ensure_src_on_path()

from pmarlo.conformations import (  # noqa: E402
    find_conformations,
    plot_committors,
    plot_flux_network,
    plot_pathways,
    plot_tpt_summary,
)
from pmarlo.conformations.visualizations import (  # noqa: E402
    plot_pcca_states,
    plot_pcca_states_on_fes,
)
from pmarlo.data.aggregate import load_shards_as_dataset  # noqa: E402
from pmarlo.markov_state_model._msm_utils import build_simple_msm  # noqa: E402
from pmarlo.markov_state_model.clustering import cluster_microstates  # noqa: E402
from pmarlo.markov_state_model.free_energy import FESResult  # noqa: E402
from pmarlo.markov_state_model.reduction import reduce_features  # noqa: E402
from pmarlo.utils.dbscan import estimate_dbscan_eps  # noqa: E402
from pmarlo.utils.path_utils import ensure_directory  # noqa: E402
from pmarlo.utils.thermodynamics import kT_kJ_per_mol  # noqa: E402

# Analysis configuration (mirrors the Streamlit UI selections)
LAG_STEPS = 110
TICA_COMPONENTS = 5
TEMPERATURE_K = 300.0
N_METASTABLE = 8
MAX_PATHS = 8
BOOTSTRAP_SAMPLES = 50
DBSCAN_MIN_SAMPLES = 5
DBSCAN_LEAF_SIZE = 30
MIN_KIS_STATES = 10


def _json_default(value: object) -> object:
    """JSON serializer that handles numpy scalars and complex values."""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, complex) or hasattr(value, "real") and hasattr(value, "imag"):
        real = float(value.real)
        imag = float(value.imag)
        if abs(imag) < 1e-12:
            return real
        return {"real": real, "imag": imag}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _compute_tica_fes(
    features: np.ndarray,
    *,
    temperature: float,
    bins: int = 72,
) -> FESResult:
    """Compute a simple 2D free-energy surface over the first two TICA components."""

    if features.shape[1] < 2:
        raise ValueError("At least two TICA components are required to plot a FES.")
    coords = np.asarray(features[:, :2], dtype=float)
    x = coords[:, 0]
    y = coords[:, 1]
    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    counts = counts.T  # Align orientation with plotting utilities
    kT = kT_kJ_per_mol(temperature)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = counts / np.sum(counts)
        F = np.full_like(prob, np.inf, dtype=float)
        mask = prob > 0
        F[mask] = -kT * np.log(prob[mask])
    finite_mask = np.isfinite(F)
    if np.any(finite_mask):
        F[finite_mask] -= np.min(F[finite_mask])
    metadata = {
        "counts": counts,
        "temperature": temperature,
        "cv1_name": "TICA1",
        "cv2_name": "TICA2",
    }
    return FESResult(F=F, xedges=xedges, yedges=yedges, metadata=metadata)


def _write_conformations_summary(conf_result, path: Path) -> None:
    """Write a CSV summarising each conformation."""

    header = [
        "type",
        "state_id",
        "macrostate_id",
        "population",
        "free_energy_kJ_mol",
        "committor",
        "kis_score",
        "flux",
        "structure_path",
    ]
    lines = [",".join(header)]
    for conf in conf_result.conformations:
        line = [
            conf.conformation_type,
            str(conf.state_id),
            "" if conf.macrostate_id is None else str(conf.macrostate_id),
            f"{conf.population:.10f}",
            f"{conf.free_energy:.6f}",
            "" if conf.committor is None else f"{float(conf.committor):.6f}",
            "" if conf.kis_score is None else f"{float(conf.kis_score):.6f}",
            "" if conf.flux is None else f"{float(conf.flux):.6e}",
            "" if conf.structure_path is None else str(conf.structure_path),
        ]
        lines.append(",".join(line))
    path.write_text("\n".join(lines))

EXAMPLE_ROOT = Path(__file__).resolve().parent
DEFAULT_SHARDS_ROOT = project_root() / "pmarlo_webapp" / "app_output" / "shards"
DEFAULT_TOPOLOGY = project_root() / "pmarlo_webapp" / "app_input" / "3gd8-fixed.pdb"
DEFAULT_OUTPUT_ROOT = EXAMPLE_ROOT / "programs_outputs" / "conformations"


def _collect_shard_paths(shards_root: Path) -> list[Path]:
    """Return every shard JSON emitted by the webapp."""

    if not shards_root.is_dir():
        raise FileNotFoundError(f"Shard root {shards_root} does not exist.")
    json_paths = sorted(shards_root.glob("run-*/**/*.json"))
    json_paths = [path for path in json_paths if path.is_file() and "manifest" not in path.name]
    if not json_paths:
        raise RuntimeError(f"No shard JSON files were found underneath {shards_root}.")
    return [path.resolve() for path in json_paths]


def _summarize_selection(paths: Sequence[Path]) -> None:
    """Print the run/shard selection summary."""

    run_names = sorted({path.parent.name for path in paths})
    print(f"Runs selected {len(run_names)} / {len(run_names)}")
    print(f"Shards selected {len(paths)} / {len(paths)}")


def _prepare_output_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target = root / f"dbscan_conformations_{timestamp}"
    ensure_directory(target)
    return target


def run_conformations_example(
    shards_root: Path,
    topology_pdb: Path,
    output_root: Path,
) -> Path:
    """Execute the DBSCAN-based conformations analysis over all shards."""

    shard_paths = _collect_shard_paths(shards_root)
    _summarize_selection(shard_paths)

    ensure_directory(output_root)
    output_dir = _prepare_output_dir(output_root)
    plots_dir = output_dir / "plots"
    ensure_directory(plots_dir)

    if not topology_pdb.is_file():
        raise FileNotFoundError(f"Topology PDB {topology_pdb} was not found.")

    print("\n[1/6] Loading shards into memory")
    dataset = load_shards_as_dataset(shard_paths)
    features = np.asarray(dataset.get("X", []), dtype=float)
    if features.size == 0 or features.ndim != 2:
        raise ValueError("Aggregated dataset did not contain feature matrix 'X'.")

    total_frames = int(features.shape[0])
    print(f"Frames selected {total_frames:,} / {total_frames:,}")
    if features.shape[1] < TICA_COMPONENTS:
        raise ValueError(
            f"Dataset provides only {features.shape[1]} features; "
            f"{TICA_COMPONENTS} TICA components were requested."
        )

    print(f"\n[2/6] Reducing features with TICA (lag={LAG_STEPS}, components={TICA_COMPONENTS})")
    features_reduced = reduce_features(
        features,
        method="tica",
        lag=LAG_STEPS,
        n_components=TICA_COMPONENTS,
    )
    print(f"Reduced dimensionality: {features_reduced.shape[1]} components")
    features_reduced_full = np.asarray(features_reduced, dtype=float)
    fes_result = _compute_tica_fes(
        features_reduced_full,
        temperature=TEMPERATURE_K,
    )

    print("\n[3/6] Clustering with DBSCAN (auto eps seed)")
    base_eps, eps_meta = estimate_dbscan_eps(
        features_reduced_full,
        min_samples=DBSCAN_MIN_SAMPLES,
        random_state=None,
    )
    eps_factors = (1.0, 0.9, 0.8, 0.7, 0.6)
    required_microstates = max(N_METASTABLE, MIN_KIS_STATES)
    clustering_info: dict[str, float | int | str | np.ndarray] | None = None

    for factor in eps_factors:
        eps_value = float(base_eps * factor)
        print(f"  -> Attempting eps={eps_value:.6f} (factor {factor:.2f})")
        clustering = cluster_microstates(
            features_reduced_full,
            method="dbscan",
            n_states="auto",
            random_state=None,
            min_samples=DBSCAN_MIN_SAMPLES,
            leaf_size=DBSCAN_LEAF_SIZE,
            eps=eps_value,
        )
        labels_raw = np.asarray(clustering.labels, dtype=int).reshape(-1)
        if labels_raw.shape[0] != features_reduced_full.shape[0]:
            raise ValueError("Label array shape does not match feature matrix rows.")

        noise_frames = int(np.count_nonzero(labels_raw < 0))
        if noise_frames:
            print(f"     DBSCAN flagged {noise_frames} frames as noise (label -1)")

        valid_mask = labels_raw >= 0
        effective_frames = int(np.count_nonzero(valid_mask))
        if effective_frames == 0:
            print("     All frames were labelled noise; reducing eps further.")
            continue

        labels_filtered = labels_raw[valid_mask]
        unique_states = np.unique(labels_filtered)
        n_states = int(unique_states.size)
        if n_states < 2:
            print("     Produced <2 microstates; reducing eps further.")
            continue

        if n_states < required_microstates:
            print(
                f"     Produced {n_states} microstates (< {required_microstates}); "
                "reducing eps further."
            )
            continue

        remap = {state: idx for idx, state in enumerate(unique_states)}
        labels_dense = np.asarray([remap[int(label)] for label in labels_filtered], dtype=int)
        features_filtered = features_reduced_full[valid_mask]
        cluster_centers = np.zeros((n_states, features_filtered.shape[1]), dtype=float)
        for state_idx in range(n_states):
            state_mask = labels_dense == state_idx
            if not np.any(state_mask):
                continue
            cluster_centers[state_idx] = features_filtered[state_mask].mean(axis=0)

        clustering_info = {
            "labels": labels_dense,
            "features": features_filtered,
            "noise_frames": noise_frames,
            "effective_frames": effective_frames,
            "n_states": n_states,
            "eps": eps_value,
            "rationale": clustering.rationale or "dbscan",
            "centers": cluster_centers,
        }
        break

    if clustering_info is None:
        raise ValueError(
            "DBSCAN could not identify enough clusters even after tightening eps."
        )

    labels = np.asarray(clustering_info["labels"], dtype=int)
    features_reduced = np.asarray(clustering_info["features"], dtype=float)
    noise_frames = int(clustering_info["noise_frames"])
    effective_frames = int(clustering_info["effective_frames"])
    n_states = int(clustering_info["n_states"])
    eps_used = float(clustering_info["eps"])
    rationale = str(clustering_info["rationale"])

    print(f"Effective frames after removing noise: {effective_frames:,}")
    print(f"Discovered {n_states} microstates ({rationale}, eps={eps_used:.6f})")

    print(f"\n[4/6] Building MSM at lag={LAG_STEPS}")
    T, pi = build_simple_msm(
        [labels],
        n_states=n_states,
        lag=LAG_STEPS,
        count_mode="sliding",
    )
    print(f"Transition matrix shape: {T.shape}, stationary sum={np.sum(pi):.6f}")

    print("\n[5/6] Running TPT conformations analysis")
    msm_data = {
        "T": T,
        "pi": pi,
        "dtrajs": [labels],
        "features": features_reduced,
    }
    conf_result = find_conformations(
        msm_data=msm_data,
        trajectories=None,
        auto_detect=True,
        auto_detect_method="auto",
        find_transition_states=True,
        find_metastable_states=True,
        find_pathway_intermediates=True,
        compute_kis=True,
        uncertainty_analysis=True,
        n_bootstrap=BOOTSTRAP_SAMPLES,
        representative_selection="medoid",
        output_dir=str(output_dir / "representatives"),
        save_structures=False,
        n_metastable=N_METASTABLE,
        n_paths=MAX_PATHS,
        temperature=TEMPERATURE_K,
        topology_path=str(topology_pdb),
    )

    macro_memberships = np.asarray(
        conf_result.metadata.get("macrostate_memberships", []),
        dtype=float,
    )
    pcca_plot_paths: dict[str, str] = {}
    if macro_memberships.ndim == 2 and macro_memberships.shape[0] == n_states:
        cluster_centers = np.asarray(clustering_info["centers"], dtype=float)
        if cluster_centers.shape[1] < 2:
            raise ValueError("Cluster centers must include at least two TICA dimensions.")
        tica_cluster_coords = cluster_centers[:, :2]
        pcca_states_path = plots_dir / "pcca_states.png"
        plot_pcca_states(
            tica_cluster_coords,
            macro_memberships,
            str(pcca_states_path),
            xlabel="TICA 1",
            ylabel="TICA 2",
        )
        pcca_plot_paths["pcca_states"] = str(pcca_states_path)
        if fes_result is not None:
            pcca_fes_path = plots_dir / "pcca_states_on_fes.png"
            plot_pcca_states_on_fes(
                fes_result,
                tica_cluster_coords,
                macro_memberships,
                str(pcca_fes_path),
            )
            pcca_plot_paths["pcca_states_on_fes"] = str(pcca_fes_path)

    _write_conformations_summary(conf_result, output_dir / "conformations_summary.csv")

    print("\n[6/6] Writing artefacts")
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shards_root": str(shards_root),
        "topology_pdb": str(topology_pdb),
        "lag_steps": LAG_STEPS,
        "tica_components": TICA_COMPONENTS,
        "dbscan": {
            "min_samples": DBSCAN_MIN_SAMPLES,
            "leaf_size": DBSCAN_LEAF_SIZE,
            "noise_frames": noise_frames,
            "clusters": n_states,
            "eps": eps_used,
            "estimation_percentile": float(eps_meta["percentile"]) if eps_meta else None,
            "sample_size": float(eps_meta["sample_size"]) if eps_meta else None,
            "neighbor_rank": float(eps_meta["neighbor_rank"]) if eps_meta else None,
            "rationale": rationale,
        },
        "n_metastable": N_METASTABLE,
        "n_paths": MAX_PATHS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "temperature_K": TEMPERATURE_K,
        "frames_total": total_frames,
        "frames_effective": effective_frames,
        "pcca_plots": pcca_plot_paths,
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(summary, indent=2))
    conformations_path = output_dir / "conformations.json"
    conformations_path.write_text(
        json.dumps(conf_result.to_dict(), indent=2, default=_json_default)
    )

    if conf_result.tpt_result:
        plot_tpt_summary(conf_result.tpt_result, str(plots_dir))
        plot_committors(
            conf_result.tpt_result.forward_committor,
            conf_result.tpt_result.backward_committor,
            output_path=str(plots_dir / "committors.png"),
        )
        plot_flux_network(
            conf_result.tpt_result.flux_matrix,
            conf_result.tpt_result.net_flux,
            source_states=conf_result.tpt_result.source_states.tolist(),
            sink_states=conf_result.tpt_result.sink_states.tolist(),
            output_path=str(plots_dir / "flux_network.png"),
        )
        if conf_result.tpt_result.pathways:
            plot_pathways(
                conf_result.tpt_result.pathways,
                conf_result.tpt_result.pathway_fluxes,
                str(plots_dir / "pathways.png"),
            )

    print(f"\nAnalysis complete. Outputs written to: {output_dir}")
    return output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DBSCAN-based conformations analysis on all shards emitted by the webapp."
        )
    )
    parser.add_argument(
        "--shards-root",
        type=Path,
        default=DEFAULT_SHARDS_ROOT,
        help="Root directory containing run-* shard folders "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        default=DEFAULT_TOPOLOGY,
        help="Topology PDB used for representative extraction "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that will store the analysis artefacts "
        "(default: %(default)s)",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_conformations_example(
        shards_root=args.shards_root,
        topology_pdb=args.topology,
        output_root=args.output_dir,
    )


if __name__ == "__main__":
    main()
