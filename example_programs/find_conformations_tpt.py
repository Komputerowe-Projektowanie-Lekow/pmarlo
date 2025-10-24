"""Example: Finding protein conformations using TPT analysis.

This example demonstrates the complete workflow for identifying and analyzing
protein conformations using Transition Path Theory (TPT), including:
- Auto-detection of source/sink states
- TPT analysis (committors, reactive flux, pathways)
- Kinetic Importance Score (KIS) computation
- Identification of transition states, metastable states, and pathway intermediates
- Representative structure extraction and saving
- Uncertainty quantification via bootstrap

Outputs saved in: example_programs/programs_outputs/conformations_tpt/
"""

from __future__ import annotations

from pathlib import Path

from _example_support import assets_path, ensure_src_on_path

ensure_src_on_path()

import mdtraj as md
import numpy as np

from pmarlo import api
from pmarlo.conformations import find_conformations
from pmarlo.utils.path_utils import ensure_directory

# Configuration
ASSETS_DIR = assets_path()
PDB_PATH = ASSETS_DIR / "3gd8-fixed.pdb"
DCD_PATH = ASSETS_DIR / "traj.dcd"

OUT_DIR = (
    Path(__file__).resolve().parent / "programs_outputs" / "conformations_tpt"
)
ensure_directory(OUT_DIR)

# Analysis parameters
FEATURE_SPEC = ["phi_psi"]
LAG_TIME = 10
N_CLUSTERS = 20
N_COMPONENTS = 3
N_METASTABLE = 4
TEMPERATURE_K = 300.0


def main():
    """Run complete conformations analysis."""
    print("=" * 70)
    print("Conformations Finder with Transition Path Theory")
    print("=" * 70)

    # Step 1: Load trajectory
    print(f"\n[1/8] Loading trajectory from {DCD_PATH}")
    traj = md.load(str(DCD_PATH), top=str(PDB_PATH))
    print(f"  Loaded {len(traj)} frames")

    # Step 2: Compute features
    print(f"\n[2/8] Computing features: {FEATURE_SPEC}")
    features, cols, periodic = api.compute_features(traj, feature_specs=FEATURE_SPEC)
    print(f"  Features shape: {features.shape}")

    # Step 3: Dimensionality reduction with TICA
    print(f"\n[3/8] Reducing dimensionality with TICA (lag={LAG_TIME})")
    features_reduced = api.reduce_features(
        features, method="tica", lag=LAG_TIME, n_components=N_COMPONENTS
    )
    print(f"  Reduced to {N_COMPONENTS} dimensions")

    # Step 4: Clustering
    print(f"\n[4/8] Clustering into {N_CLUSTERS} microstates")
    labels = api.cluster_microstates(
        features_reduced,
        method="minibatchkmeans",
        n_states=N_CLUSTERS,
        n_init=50,
    )
    n_states = int(np.max(labels) + 1)
    print(f"  Created {n_states} microstates")

    # Step 5: Build MSM
    print(f"\n[5/8] Building MSM (lag={LAG_TIME})")
    T, pi = api.build_simple_msm(
        [labels], n_states=n_states, lag=LAG_TIME, count_mode="sliding"
    )
    print(f"  Transition matrix shape: {T.shape}")
    print(f"  Stationary distribution sum: {np.sum(pi):.6f}")

    # Compute implied timescales for k_slow selection
    print("  Computing implied timescales...")
    its = []
    for k in range(2, min(n_states, 10)):
        eigenvalues = np.linalg.eigvals(T)
        sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
        if k < len(sorted_eigs):
            # ITS = -lag / ln(lambda_k)
            if sorted_eigs[k] > 0:
                its.append(-LAG_TIME / np.log(sorted_eigs[k]))
    its_array = np.array(its) if its else None

    if its_array is not None:
        print(f"  Implied timescales (top 5): {its_array[:5]}")

    # Step 6: Run conformations finder with TPT
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
        trajectories=traj,
        auto_detect=True,
        auto_detect_method="auto",
        find_transition_states=True,
        find_metastable_states=True,
        find_pathway_intermediates=True,
        compute_kis=True,
        uncertainty_analysis=True,
        n_bootstrap=100,
        representative_selection="medoid",
        output_dir=str(OUT_DIR),
        save_structures=True,
        temperature=TEMPERATURE_K,
        lag=LAG_TIME,
        n_metastable=N_METASTABLE,
        n_paths=5,
        k_slow="auto",
    )

    # Step 7: Display results
    print(f"\n[7/8] Analysis complete!")
    print("  " + "-" * 66)

    # TPT results
    if results.tpt_result:
        tpt = results.tpt_result
        print(f"  TPT Analysis:")
        print(f"    Source states: {len(tpt.source_states)}")
        print(f"    Sink states: {len(tpt.sink_states)}")
        print(f"    Transition rate: {tpt.rate:.6e}")
        print(f"    Mean first passage time: {tpt.mfpt:.2f} steps")
        print(f"    Total reactive flux: {tpt.total_flux:.6e}")
        print(f"    Number of pathways: {len(tpt.pathways)}")
        if len(tpt.pathways) > 0:
            print(f"    Top pathway length: {len(tpt.pathways[0])} states")

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

    # Uncertainty results
    if results.uncertainty_results:
        print(f"\n  Uncertainty quantification:")
        for unc in results.uncertainty_results[:3]:  # Show first 3
            print(f"    {unc.observable_name}:")
            if isinstance(unc.mean, float):
                print(f"      Mean: {unc.mean:.3e} ± {unc.std:.3e}")
                print(f"      95% CI: [{unc.ci_lower:.3e}, {unc.ci_upper:.3e}]")

    # Step 8: Save results
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

    # Print metastable states details
    if metastable:
        print(f"\n  Metastable States (saved in {OUT_DIR}/metastable/):")
        for conf in metastable:
            print(
                f"    Macrostate {conf.macrostate_id}: "
                f"State {conf.state_id}, pop={conf.population:.4f}, "
                f"F={conf.free_energy:.2f} kJ/mol"
            )

    # Print transition states details
    if transition:
        print(f"\n  Transition States (saved in {OUT_DIR}/transition/):")
        for conf in transition[:5]:  # Show first 5
            print(
                f"    State {conf.state_id}: "
                f"committor={conf.committor:.3f}, "
                f"pop={conf.population:.4f}"
            )

    print("\n" + "=" * 70)
    print("Analysis complete! Check output directory for structures and results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
