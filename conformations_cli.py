#!/usr/bin/env python
"""CLI tool for running conformations analysis on shards.

This mimics the Streamlit app button functionality but runs from command line.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Run TPT conformations analysis on shards"
    )
    parser.add_argument(
        "--shards-dir",
        type=Path,
        default=Path("example_programs/app_usecase/app_intputs/experiments/mixed_ladders_shards"),
        help="Directory containing shard files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example_programs/programs_outputs/conformations_cli"),
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
        "--shard-indices",
        type=str,
        default="all",
        help="Comma-separated shard indices or 'all'",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TPT CONFORMATIONS ANALYSIS - CLI")
    print("=" * 70)

    # Find shard files
    if not args.shards_dir.exists():
        print(f"\nError: Shards directory not found: {args.shards_dir}")
        print("\nAvailable directories:")
        parent = args.shards_dir.parent
        if parent.exists():
            for d in parent.iterdir():
                if d.is_dir():
                    print(f"  - {d}")
        return 1

    shard_files = sorted(args.shards_dir.glob("*.json"))
    shard_files = [f for f in shard_files if not f.name.startswith("manifest")]

    if not shard_files:
        print(f"\nError: No shard files found in {args.shards_dir}")
        return 1

    print(f"\nFound {len(shard_files)} shard files in {args.shards_dir}")

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
    topology_pdb = None
    shard_meta_list = dataset.get("__shards__", [])
    for shard_meta in shard_meta_list:
        if isinstance(shard_meta, dict):
            pdb_path = shard_meta.get("structure_pdb")
            if pdb_path:
                topology_pdb = Path(pdb_path)
                if topology_pdb.exists():
                    print(f"  [OK] Found topology: {topology_pdb}")
                    break

    if topology_pdb is None or not topology_pdb.exists():
        print("  [WARN] No topology PDB found, will skip trajectory loading")
        combined_traj = None
    else:
        # Load trajectories
        print("\n[4/8] Loading trajectories...")
        all_trajs = []
        for shard_meta in shard_meta_list:
            if isinstance(shard_meta, dict):
                traj_paths = shard_meta.get("trajectories", [])
                for traj_path_str in traj_paths:
                    traj_path = Path(traj_path_str)
                    if traj_path.exists():
                        try:
                            traj = md.load(str(traj_path), top=str(topology_pdb))
                            all_trajs.append(traj)
                            print(f"  [OK] Loaded: {traj_path.name} ({len(traj)} frames)")
                        except Exception as e:
                            print(f"  [WARN] Failed to load {traj_path.name}: {e}")

        combined_traj = md.join(all_trajs) if all_trajs else None
        if combined_traj:
            print(f"  [OK] Combined {len(combined_traj)} frames")

    # Dimensionality reduction
    print(f"\n[5/8] Reducing features with TICA (lag={args.lag}, n_components={args.n_components})...")
    try:
        features_reduced = reduce_features(
            features, method="tica", lag=args.lag, n_components=args.n_components
        )
        print(f"  [OK] Reduced to {features_reduced.shape[1]} dimensions")
    except Exception as e:
        print(f"Error in dimensionality reduction: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Clustering
    print(f"\n[6/8] Clustering into {args.n_clusters} microstates...")
    try:
        clustering_result = cluster_microstates(
            features_reduced,
            method="minibatchkmeans",
            n_states=args.n_clusters,
            random_state=42,
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

    try:
        # Prepare MSM data dictionary
        msm_data = {
            'T': T,
            'pi': pi,
            'dtrajs': [labels],
            'features': features_reduced,
        }
        
        conf_result = find_conformations(
            msm_data=msm_data,
            trajectories=[combined_traj] if combined_traj else None,
            auto_detect=True,
            auto_detect_method='auto',
            find_transition_states=True,
            find_metastable_states=True,
            find_pathway_intermediates=True,
            compute_kis=True,
            uncertainty_analysis=False,
            n_bootstrap=50,
            representative_selection='medoid',
            output_dir=str(args.output_dir),
            save_structures=True,
        )
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
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if conf_result.tpt_result:
        print(f"\nTPT Metrics:")
        print(f"  Rate:        {conf_result.tpt_result.rate:.3e} / step")
        print(f"  MFPT:        {conf_result.tpt_result.mfpt:.1f} steps")
        print(f"  Total Flux:  {conf_result.tpt_result.total_flux:.3e}")
        print(f"  N Pathways:  {len(conf_result.tpt_result.pathways)}")
        print(f"  Source:      {len(conf_result.tpt_result.source_states)} states")
        print(f"  Sink:        {len(conf_result.tpt_result.sink_states)} states")

    metastable = conf_result.get_by_type('metastable')
    if metastable:
        print(f"\nMetastable States: {len(metastable)}")
        for conf in metastable[:5]:
            print(f"  {conf.state_id}: pop={conf.population:.4f}")

    transition = conf_result.get_transition_states()
    if transition:
        print(f"\nTransition States: {len(transition)}")
        for conf in transition[:5]:
            print(f"  State {conf.state_id}: committor={conf.committor:.3f}")

    # Count PDB files
    pdb_files = list(args.output_dir.glob("*.pdb"))
    print(f"\nOutput Directory: {args.output_dir}")
    print(f"  PDB files: {len(pdb_files)}")
    print(f"  Plot files: {len(list(args.output_dir.glob('*.png')))}")

    # Save summary JSON
    summary_path = args.output_dir / "conformations_summary.json"
    
    metastable = conf_result.get_by_type('metastable')
    transition = conf_result.get_transition_states()
    
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
            "n_components": args.n_components,
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

