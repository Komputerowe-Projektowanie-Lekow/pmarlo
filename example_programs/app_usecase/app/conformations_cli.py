#!/usr/bin/env python
"""CLI tool for running conformations analysis on shards.

This mimics the Streamlit app button functionality but runs from command line.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run TPT conformations analysis on shards"
    )
    parser.add_argument(
        "--shards-dir",
        type=Path,
        default=Path("../app_intputs/experiments/mixed_ladders_shards"),
        help="Directory containing shard files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../programs_outputs/conformations_cli"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--lag", type=int, default=10, help="Lag time for MSM"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=30, help="Number of microstates"
    )
    parser.add_argument(
        "--n-components", type=int, default=3, help="TICA components"
    )
    parser.add_argument(
        "--n-metastable", type=int, default=4, help="Number of metastable states"
    )
    parser.add_argument(
        "--temperature", type=float, default=300.0, help="Temperature in Kelvin"
    )
    parser.add_argument(
        "--n-paths", type=int, default=10, help="Maximum number of pathways"
    )
    parser.add_argument(
        "--uncertainty-analysis",
        action="store_true",
        help="Enable bootstrap uncertainty quantification during TPT analysis.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=50,
        help="Number of bootstrap samples used when uncertainty analysis is enabled.",
    )
    parser.add_argument(
        "--cluster-mode",
        type=str,
        default="kmeans",
        choices=["kmeans", "minibatchkmeans", "auto"],
        help="Clustering algorithm to use for microstate assignment",
    )
    parser.add_argument(
        "--cluster-seed",
        type=int,
        default=42,
        help="Random seed for clustering (-1 disables deterministic seeding)",
    )
    parser.add_argument(
        "--kmeans-n-init",
        type=int,
        default=50,
        help=(
            "Number of random initialisations evaluated when clustering microstates."
        ),
    )
    parser.add_argument(
        "--kmeans-kwargs",
        type=str,
        default=None,
        help=(
            "JSON object with additional parameters forwarded to the KMeans-based "
            "clusterers (e.g. '{\"max_iter\": 200}')"
        ),
    )
    parser.add_argument(
        "--shard-indices",
        type=str,
        default="all",
        help="Comma-separated shard indices or 'all'",
    )
    parser.add_argument(
        "--topology",
        type=Path,
        required=True,
        help="Path to the topology PDB file corresponding to the trajectories.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TPT CONFORMATIONS ANALYSIS - CLI")
    print("=" * 70)

    if args.kmeans_kwargs:
        try:
            kmeans_kwargs = json.loads(args.kmeans_kwargs)
        except json.JSONDecodeError as exc:
            print(f"Error: Failed to parse --kmeans-kwargs JSON: {exc}")
            return 1
        if not isinstance(kmeans_kwargs, dict):
            print("Error: --kmeans-kwargs must decode to a JSON object")
            return 1
    else:
        kmeans_kwargs = {}

    if args.kmeans_n_init <= 0:
        print("Error: --kmeans-n-init must be a positive integer")
        return 1

    cluster_seed = None if args.cluster_seed < 0 else int(args.cluster_seed)
    cluster_mode = args.cluster_mode.strip().lower()
    method_alias = {
        "kmeans": "kmeans",
        "minibatchkmeans": "minibatchkmeans",
        "auto": "auto",
    }
    if cluster_mode not in method_alias:
        print(f"Error: Unsupported cluster-mode '{args.cluster_mode}'")
        return 1
    cluster_method = method_alias[cluster_mode]

    # Find shard files
    shards_dir = args.shards_dir.expanduser().resolve()
    if not shards_dir.exists():
        print(f"\nError: Shards directory not found: {shards_dir}")
        print("\nAvailable directories:")
        parent = shards_dir.parent
        if parent.exists():
            for d in parent.iterdir():
                if d.is_dir():
                    print(f"  - {d}")
        return 1

    shard_files = sorted(shards_dir.glob("*.json"))
    shard_files = [f for f in shard_files if not f.name.startswith("manifest")]

    if not shard_files:
        print(f"\nError: No shard files found in {shards_dir}")
        return 1

    print(f"\nFound {len(shard_files)} shard files in {shards_dir}")

    # Select shards
    if args.shard_indices == "all":
        selected_shards = shard_files
    else:
        indices = [int(i.strip()) for i in args.shard_indices.split(",")]
        selected_shards = [shard_files[i] for i in indices if i < len(shard_files)]

    print(f"Using {len(selected_shards)} shards for analysis")

    # Import modules
    print("\n[1/8] Importing modules...")
    try:
        from pmarlo.conformations import find_conformations
        from pmarlo.conformations.visualizations import (
            plot_tpt_summary,
            plot_committors,
            plot_flux_network,
            plot_pathways,
        )
        from pmarlo.data.aggregate import load_shards_as_dataset
        from pmarlo.markov_state_model.bridge import build_simple_msm
        from pmarlo.markov_state_model.clustering import cluster_microstates
        from pmarlo.markov_state_model.reduction import reduce_features
        import mdtraj as md
    except ImportError as e:
        print(f"Error importing modules: {e}")
        return 1

    print("  [OK] All modules imported successfully")

    # Load shards
    print(f"\n[2/8] Loading {len(selected_shards)} shards...")
    try:
        dataset = load_shards_as_dataset(selected_shards)
        if "X" not in dataset or len(dataset["X"]) == 0:
            print("Error: No feature data found in shards")
            return 1

        features = np.asarray(dataset["X"], dtype=float)
        print(f"  [OK] Loaded {features.shape[0]} frames with {features.shape[1]} features")
    except Exception as e:
        print(f"Error loading shards: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Find topology
    print("\n[3/8] Finding topology PDB...")
    topology_pdb = args.topology.expanduser().resolve()
    if not topology_pdb.exists():
        print(f"Error: Topology PDB not found at path: {topology_pdb}")
        return 1
    print(f"  [OK] Using provided topology: {topology_pdb}")

    # Load trajectories
    print("\n[4/8] Loading trajectories...")
    shard_meta_list = dataset.get("__shards__", [])
    all_trajs = []
    for idx, shard_meta in enumerate(shard_meta_list):
        if not isinstance(shard_meta, dict):
            print(
                "Error: Shard metadata entry "
                f"{idx} is not a mapping and does not list trajectories."
            )
            return 1
        traj_paths = shard_meta.get("trajectories")
        if not traj_paths:
            print(
                "Error: Shard metadata entry "
                f"{idx} does not include trajectory paths."
            )
            return 1
        for traj_path_str in traj_paths:
            traj_path = Path(traj_path_str)
            if traj_path.is_absolute():
                traj_path = traj_path.expanduser().resolve()
            else:
                traj_path = (shards_dir / traj_path).expanduser().resolve()
            if not traj_path.exists():
                print(f"Error: Trajectory file not found: {traj_path}")
                return 1
            try:
                traj = md.load(str(traj_path), top=str(topology_pdb))
            except Exception as e:
                print(f"Error: Failed to load {traj_path.name}: {e}")
                return 1
            all_trajs.append(traj)
            print(f"  [OK] Loaded: {traj_path.name} ({len(traj)} frames)")

    combined_traj = md.join(all_trajs) if all_trajs else None
    if combined_traj is None:
        print("Error: No trajectories could be loaded from the provided shards.")
        return 1
    print(f"  [OK] Combined {len(combined_traj)} frames")

    trajectories_loaded = True

    # Dimensionality reduction
    print(f"\n[5/8] Reducing features with TICA (lag={args.lag}, n_components={args.n_components})...")
    n_features = features.shape[1]
    tica_dim = args.n_components
    if args.n_components > n_features:
        print(
            f"[WARN] Requested {args.n_components} TICA components, but input data only has {n_features} features. Setting n_components = {n_features}."
        )
        tica_dim = n_features

    try:
        features_reduced = reduce_features(
            features, method="tica", lag=args.lag, n_components=tica_dim
        )
        print(f"  [OK] Reduced to {features_reduced.shape[1]} dimensions")
    except Exception as e:
        print(f"Error in dimensionality reduction: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Clustering
    print(
        f"\n[6/8] Clustering into {args.n_clusters} microstates using {cluster_method} "
        f"(seed={'None' if cluster_seed is None else cluster_seed}, n_init={args.kmeans_n_init})..."
    )
    try:
        clustering_result = cluster_microstates(
            features_reduced,
            method=cluster_method,
            n_states=args.n_clusters,
            random_state=cluster_seed,
            n_init=args.kmeans_n_init,
            **kmeans_kwargs,
        )
        # Extract labels from ClusteringResult object
        labels = clustering_result.labels
        n_states = int(np.max(labels) + 1)
        print(f"  [OK] Created {n_states} microstates")
    except Exception as e:
        print(f"Error in clustering: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Build MSM
    print(f"\n[7/8] Building MSM (lag={args.lag})...")
    try:
        T, pi = build_simple_msm(
            [labels], n_states=n_states, lag=args.lag, count_mode="sliding"
        )
        print(f"  [OK] Transition matrix shape: {T.shape}")
        print(f"  [OK] Stationary distribution sum: {np.sum(pi):.6f}")
    except Exception as e:
        print(f"Error building MSM: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run TPT conformations analysis
    print(f"\n[8/8] Running TPT conformations analysis...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pathway_warning = False

    try:
        # Prepare MSM data dictionary
        msm_data = {
            'T': T,
            'pi': pi,
            'dtrajs': [labels],
            'features': features_reduced,
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            conf_result = find_conformations(
                msm_data=msm_data,
                trajectories=[combined_traj] if trajectories_loaded else None,
                auto_detect=True,
                auto_detect_method='auto',
                find_transition_states=True,
                find_metastable_states=True,
                find_pathway_intermediates=True,
                compute_kis=True,
                uncertainty_analysis=args.uncertainty_analysis,
                n_bootstrap=args.bootstrap_samples,
                representative_selection='medoid',
                output_dir=str(args.output_dir),
                save_structures=trajectories_loaded,
                n_metastable=args.n_metastable,  # Pass the n_metastable parameter
            )

            for warning in w:
                if issubclass(warning.category, RuntimeWarning) and "Maximum number of iterations reached" in str(warning.message):
                    pathway_warning = True
        print("  [OK] TPT analysis complete")
    except Exception as e:
        print(f"Error in TPT analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Generate visualizations
    print("\n[9/9] Generating visualizations...")
    try:
        if conf_result.tpt_result:
            # TPT summary
            plot_tpt_summary(conf_result.tpt_result, str(args.output_dir))
            print(f"  [OK] Saved TPT summary plot")

            # Committors
            plot_committors(
                conf_result.tpt_result.forward_committor,
                conf_result.tpt_result.backward_committor,
                str(args.output_dir / "committors.png"),
            )
            print(f"  [OK] Saved committor plots")

            # Flux network
            plot_flux_network(
                conf_result.tpt_result.net_flux,
                str(args.output_dir / "flux_network.png"),
            )
            print(f"  [OK] Saved flux network")

            # Pathways
            if conf_result.tpt_result.pathways:
                plot_pathways(
                    conf_result.tpt_result.pathways,
                    conf_result.tpt_result.pathway_fluxes,
                    str(args.output_dir / "pathways.png"),
                )
                print(f"  [OK] Saved pathway plots")
    except Exception as e:
        print(f"Warning: Visualization error: {e}")

    # Print results
    print("\n[INFO] Printing results summary...")
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    tpt_result = conf_result.tpt_result
    metastable_states = conf_result.get_metastable_states()
    transition_states = conf_result.get_transition_states()

    if tpt_result:
        print(f"\nTPT Metrics:")
        print(f"  Rate:        {tpt_result.rate:.3e} / step")
        print(f"  MFPT:        {tpt_result.mfpt:.1f} steps")
        print(f"  Total Flux:  {tpt_result.total_flux:.3e}")
        print(f"  N Pathways:  {len(tpt_result.pathways)}")
        print(f"  Source:      {len(tpt_result.source_states)} states")
        print(f"  Sink:        {len(tpt_result.sink_states)} states")
        if pathway_warning:
            print(
                "  [WARN] Pathway decomposition did not converge (maximum iterations reached). Results may be incomplete."
            )

        identified_metastable = len(metastable_states)
        if args.n_metastable != identified_metastable:
            print(
                "[INFO] Requested "
                f"{args.n_metastable} metastable states, but PCCA+ analysis identified "
                f"{identified_metastable} robust states."
            )

    if metastable_states:
        print(f"\nMetastable States: {len(metastable_states)}")
        for conf in metastable_states:
            print(f"  {conf.state_id}: pop={conf.population:.4f}")

    if transition_states:
        if not tpt_result:
            raise RuntimeError(
                "Transition state summary requires TPT results, but none were found."
            )
        print(f"\nTransition States: {len(transition_states)}")
        for conf in transition_states:
            committor = float(tpt_result.forward_committor[conf.state_id])
            print(f"  State {conf.state_id}: committor={committor:.3f}")

    # Count PDB files
    pdb_files = list(args.output_dir.glob("*.pdb"))
    print(f"\nOutput Directory: {args.output_dir}")
    print(f"  PDB files: {len(pdb_files)}")
    print(f"  Plot files: {len(list(args.output_dir.glob('*.png')))}")

    # Save summary JSON
    summary_path = args.output_dir / "conformations_summary.json"

    metastable = metastable_states
    transition = transition_states

    summary = {
        "tpt": {
            "rate": float(conf_result.tpt_result.rate) if conf_result.tpt_result else None,
            "mfpt": float(conf_result.tpt_result.mfpt) if conf_result.tpt_result else None,
            "total_flux": float(conf_result.tpt_result.total_flux) if conf_result.tpt_result else None,
            "n_pathways": len(conf_result.tpt_result.pathways) if conf_result.tpt_result else 0,
        },
        "n_metastable_states": len(metastable),
        "n_transition_states": len(transition),
        "n_conformations": len(conf_result.conformations),
        "config": {
            "lag": args.lag,
            "n_clusters": args.n_clusters,
            "cluster_mode": args.cluster_mode,
            "cluster_seed": cluster_seed,
            "kmeans_kwargs": kmeans_kwargs,
            "kmeans_n_init": int(args.kmeans_n_init),
            "n_components": tica_dim,
            "n_metastable": args.n_metastable,
            "temperature": args.temperature,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Summary saved to: {summary_path}")
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
