"""Example: TPT conformations analysis on real shards data.

Uses real trajectory shards from app_usecase to perform complete TPT analysis.
User can specify which shards to analyze.

Usage:
    python find_conformations_tpt_real.py
    # Or specify shard indices:
    python find_conformations_tpt_real.py --shards 0,1,2,3

Outputs saved in: example_programs/programs_outputs/conformations_tpt_real/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _example_support import ensure_src_on_path

ensure_src_on_path()

import numpy as np

from pmarlo.conformations import find_conformations
from pmarlo.conformations.visualizations import plot_tpt_summary
from pmarlo.markov_state_model.bridge import build_simple_msm
from pmarlo.markov_state_model.clustering import cluster_microstates
from pmarlo.markov_state_model.reduction import reduce_features
from pmarlo.shards.schema import Shard
from pmarlo.utils.path_utils import ensure_directory

# Configuration
SHARDS_DIR = Path(__file__).parent / "app_usecase" / "app_intputs" / "experiments"
OUT_DIR = (
    Path(__file__).resolve().parent / "programs_outputs" / "conformations_tpt_real"
)

# Analysis parameters
LAG_TIME = 5
N_CLUSTERS = 30
N_COMPONENTS = 3
N_METASTABLE = 4
TEMPERATURE_K = 300.0


def load_shards(shard_dir: Path, shard_indices: list[int]) -> tuple[list[Shard], any]:
    """Load shards from directory.

    Args:
        shard_dir: Directory containing shard files
        shard_indices: Indices of shards to load

    Returns:
        Tuple of (shards, topology)
    """
    # Find all shard files
    json_files = sorted(shard_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("manifest")]

    if len(json_files) == 0:
        raise FileNotFoundError(f"No shard files found in {shard_dir}")

    print(f"Found {len(json_files)} shard files in {shard_dir}")
    print(f"Available shards: {list(range(len(json_files)))}")

    # Select shards
    if not shard_indices:
        shard_indices = list(range(len(json_files)))

    print(f"Loading shards: {shard_indices}")

    shards = []
    for idx in shard_indices:
        if idx >= len(json_files):
            print(f"Warning: Shard index {idx} out of range, skipping")
            continue

        json_path = json_files[idx]
        npz_path = json_path.with_suffix(".npz")

        # Load shard
        with open(json_path, "r") as f:
            meta = json.load(f)

        data = np.load(npz_path)

        shard = Shard(
            meta=meta,
            cvs=dict(data["cvs"].item()) if "cvs" in data else None,
            energy=data["energy"] if "energy" in data else None,
            bias=data["bias"] if "bias" in data else None,
            w_frame=data["w_frame"] if "w_frame" in data else None,
        )
        shards.append(shard)
        print(f"  Loaded shard {idx}: {json_path.name}, {shard.meta['n_frames']} frames")

    # Find topology
    pdb_path = SHARDS_DIR.parent / "3gd8-fixed.pdb"
    if not pdb_path.exists():
        pdb_path = SHARDS_DIR.parent / "3gd8-fixed_run-20251021-122220.pdb"

    return shards, pdb_path


def main(shard_dataset: str = "mixed_ladders_shards", shard_indices: list[int] | None = None):
    """Run TPT conformations analysis on real shards.

    Args:
        shard_dataset: Name of shard dataset to use
        shard_indices: Optional list of shard indices to analyze
    """
    ensure_directory(OUT_DIR)

    print("=" * 70)
    print("TPT Conformations Analysis on Real Shards")
    print("=" * 70)

    # Step 1: Load shards
    print(f"\n[1/8] Loading shards from {shard_dataset}")
    shard_dir = SHARDS_DIR / shard_dataset
    if not shard_dir.exists():
        print(f"Error: Shard directory not found: {shard_dir}")
        print(f"Available datasets:")
        for d in SHARDS_DIR.iterdir():
            if d.is_dir() and "shards" in d.name:
                print(f"  - {d.name}")
        return

    shards, pdb_path = load_shards(shard_dir, shard_indices or [])

    if len(shards) == 0:
        print("Error: No shards loaded")
        return

    print(f"Total frames across shards: {sum(s.meta['n_frames'] for s in shards)}")

    # Step 2: Extract CVs from shards
    print(f"\n[2/8] Extracting CVs from shards")

    # Concatenate CVs from all shards
    all_cvs = {}
    for shard in shards:
        if shard.cvs is None:
            print(f"Error: Shard {shard.meta['shard_id']} has no CVs")
            return

        for cv_name, cv_values in shard.cvs.items():
            if cv_name not in all_cvs:
                all_cvs[cv_name] = []
            all_cvs[cv_name].append(cv_values)

    # Stack CVs
    features_dict = {name: np.concatenate(values) for name, values in all_cvs.items()}
    cv_names = list(features_dict.keys())

    print(f"  Available CVs: {cv_names}")

    # Use first 2-3 CVs for analysis
    selected_cvs = cv_names[:min(3, len(cv_names))]
    print(f"  Using CVs for analysis: {selected_cvs}")

    features = np.column_stack([features_dict[name] for name in selected_cvs])
    print(f"  Features shape: {features.shape}")

    # Step 3: Dimensionality reduction
    print(f"\n[3/8] Reducing dimensionality with TICA (lag={LAG_TIME})")
    features_reduced = reduce_features(
        features, method="tica", lag=LAG_TIME, n_components=N_COMPONENTS
    )
    print(f"  Reduced to {N_COMPONENTS} dimensions")

    # Step 4: Clustering
    print(f"\n[4/8] Clustering into {N_CLUSTERS} microstates")
    clustering_result = cluster_microstates(
        features_reduced,
        method="minibatchkmeans",
        n_states=N_CLUSTERS,
        random_state=42,
    )
    # Extract labels from ClusteringResult object
    labels = clustering_result.labels
    n_states = int(np.max(labels) + 1)
    print(f"  Created {n_states} microstates")

    # Step 5: Build MSM
    print(f"\n[5/8] Building MSM (lag={LAG_TIME})")
    T, pi = build_simple_msm(
        [labels], n_states=n_states, lag=LAG_TIME, count_mode="sliding"
    )
    print(f"  Transition matrix shape: {T.shape}")
    print(f"  Stationary distribution sum: {np.sum(pi):.6f}")

    # Compute implied timescales
    print("  Computing implied timescales...")
    eigenvalues = np.linalg.eigvals(T)
    sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
    its = []
    for k in range(1, min(n_states, 10)):
        if k < len(sorted_eigs) and sorted_eigs[k] > 0:
            its.append(-LAG_TIME / np.log(sorted_eigs[k]))
    its_array = np.array(its) if its else None

    if its_array is not None:
        print(f"  Implied timescales (top 5): {its_array[:5]}")

    # Step 6: Run TPT conformations finder
    print(f"\n[6/8] Running TPT-based conformations analysis")

    msm_data = {
        "T": T,
        "pi": pi,
        "dtrajs": [labels],
        "features": features_reduced,
        "its": its_array,
    }

    results = find_conformations(
        msm_data=msm_data,
        trajectories=None,  # No trajectory objects available from shards
        auto_detect=True,
        auto_detect_method="auto",
        find_transition_states=True,
        find_metastable_states=True,
        find_pathway_intermediates=True,
        compute_kis=True,
        uncertainty_analysis=False,  # Skip for speed with real data
        representative_selection="medoid",
        output_dir=str(OUT_DIR),
        save_structures=False,  # No trajectory objects
        temperature=TEMPERATURE_K,
        lag=LAG_TIME,
        n_metastable=N_METASTABLE,
        n_paths=10,
        pathway_fraction=0.7,
        k_slow="auto",
    )

    # Step 7: Display results
    print(f"\n[7/8] Analysis complete!")
    print("  " + "-" * 66)

    # TPT results
    if results.tpt_result:
        tpt = results.tpt_result
        print(f"  TPT Analysis:")
        print(f"    Source states: {tpt.source_states}")
        print(f"    Sink states: {tpt.sink_states}")
        print(f"    Transition rate: {tpt.rate:.6e}")
        print(f"    Mean first passage time: {tpt.mfpt:.2f} steps")
        print(f"    Total reactive flux: {tpt.total_flux:.6e}")
        print(f"    Number of pathways: {len(tpt.pathways)}")
        if len(tpt.pathways) > 0:
            print(f"\n    Top 3 pathways:")
            for i in range(min(3, len(tpt.pathways))):
                path = tpt.pathways[i]
                flux = tpt.pathway_fluxes[i]
                fraction = flux / tpt.total_flux if tpt.total_flux > 0 else 0
                print(f"      Path {i+1}: {' → '.join(map(str, path[:6]))}{'...' if len(path) > 6 else ''}")
                print(f"               Flux: {flux:.3e} ({fraction*100:.1f}%)")

    # KIS results
    if results.kis_result:
        kis = results.kis_result
        print(f"\n  Kinetic Importance Score:")
        print(f"    Number of slow modes (k_slow): {kis.k_slow}")
        print(f"    Top 5 states by KIS:")
        for i in range(min(5, len(kis.ranked_states))):
            state_id = kis.ranked_states[i]
            score = kis.kis_scores[state_id]
            pop = pi[state_id]
            print(f"      State {state_id}: KIS={score:.6f}, pop={pop:.4f}")

    # Conformations
    print(f"\n  Conformations found: {len(results.conformations)}")
    metastable = results.get_metastable_states()
    transition = results.get_transition_states()
    pathway = results.get_pathway_intermediates()

    print(f"    Metastable states: {len(metastable)}")
    print(f"    Transition states: {len(transition)}")
    print(f"    Pathway intermediates: {len(pathway)}")

    # Step 8: Save results and visualizations
    print(f"\n[8/8] Saving results to {OUT_DIR}")

    # Save JSON
    json_path = OUT_DIR / "conformations_results.json"
    results.save(json_path)
    print(f"  Saved: {json_path}")

    # Save summary CSV
    csv_path = OUT_DIR / "conformations_summary.csv"
    with open(csv_path, "w") as f:
        f.write("type,state_id,population,free_energy,committor,kis_score,flux\n")
        for conf in results.conformations:
            f.write(
                f"{conf.conformation_type},{conf.state_id},"
                f"{conf.population:.6f},{conf.free_energy:.3f},"
                f"{conf.committor if conf.committor else ''},"
                f"{conf.kis_score if conf.kis_score else ''},"
                f"{conf.flux if conf.flux else ''}\n"
            )
    print(f"  Saved: {csv_path}")

    # Create visualizations
    print(f"\n  Creating TPT visualizations...")
    if results.tpt_result:
        plot_tpt_summary(results.tpt_result, str(OUT_DIR / "plots"))
        print(f"  Saved plots to: {OUT_DIR / 'plots'}")

    print("\n" + "=" * 70)
    print("Analysis complete! Check output directory for results and plots.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPT conformations analysis on real shards")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mixed_ladders_shards",
        help="Shard dataset to use (default: mixed_ladders_shards)",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default="",
        help="Comma-separated shard indices to analyze (default: all)",
    )

    args = parser.parse_args()

    # Parse shard indices
    if args.shards:
        shard_indices = [int(x.strip()) for x in args.shards.split(",")]
    else:
        shard_indices = None

    main(shard_dataset=args.dataset, shard_indices=shard_indices)

