#!/usr/bin/env python3
"""Example demonstrating Transition Path Theory analysis using EnhancedMSM.

This example shows how to use the TPT methods integrated directly into
the EnhancedMSM class, following deeptime's API exactly.

The example uses a simple drunkards walk system to demonstrate:
1. Computing reactive flux between source and sink states
2. Calculating committor probabilities
3. Decomposing flux into dominant pathways
4. Identifying transition state ensembles
5. Finding bottleneck states
6. Coarse-graining flux onto macrostates
"""

from __future__ import annotations

from _example_support import ensure_src_on_path, example_output_dir

ensure_src_on_path()

import numpy as np

from pmarlo.markov_state_model import MarkovStateModel


def create_simple_msm():
    """Create a simple 10-state MSM with clear metastable states."""
    # Create a linear chain with two metastable basins
    # States 0-2: source basin (high stability)
    # States 3-6: transition region
    # States 7-9: sink basin (high stability)

    T = np.array(
        [
            [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.6, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
        ]
    )

    # Compute stationary distribution
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    idx = np.argmax(np.abs(eigenvalues))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / np.sum(pi)

    return T, pi


def main():
    """Run TPT analysis example."""
    print("=" * 80)
    print("Transition Path Theory Analysis Example")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = example_output_dir("10_tpt_basic")

    # Create simple MSM
    print("Creating simple 10-state MSM...")
    T, pi = create_simple_msm()

    # Initialize MSM object
    msm = MarkovStateModel(
        trajectory_files=None,
        topology_file=None,
        output_dir=str(output_dir),
    )

    # Set transition matrix and stationary distribution
    msm.transition_matrix = T
    msm.stationary_distribution = pi
    msm.n_states = 10

    print(f"  States: {msm.n_states}")
    print(f"  Stationary distribution: {pi}")
    print()

    # Define source and sink states
    source_states = [0, 1, 2]  # Source basin
    sink_states = [7, 8, 9]  # Sink basin

    print(f"Source states (basin A): {source_states}")
    print(f"Sink states (basin B): {sink_states}")
    print()

    # 1. Compute reactive flux
    print("-" * 80)
    print("1. Computing Reactive Flux")
    print("-" * 80)

    flux = msm.reactive_flux(source_states, sink_states)

    print(f"Total flux (A→B): {flux.total_flux:.6e}")
    print(f"Transition rate k_AB: {flux.rate:.6e}")
    print(f"Mean first passage time (MFPT): {flux.mfpt:.2f} time steps")
    print()

    # 2. Compute committors
    print("-" * 80)
    print("2. Computing Committor Probabilities")
    print("-" * 80)

    q_forward = msm.compute_committor(source_states, sink_states, forward=True)
    q_backward = msm.compute_committor(source_states, sink_states, forward=False)

    print("Forward committor q+ (probability of reaching B before A):")
    for i, q in enumerate(q_forward):
        print(f"  State {i}: q+ = {q:.4f}")
    print()

    print("Backward committor q- (probability came from A rather than B):")
    for i, q in enumerate(q_backward):
        print(f"  State {i}: q- = {q:.4f}")
    print()

    # 3. Get flux matrices
    print("-" * 80)
    print("3. Flux Matrices")
    print("-" * 80)

    gross_flux = msm.get_gross_flux(source_states, sink_states)
    net_flux = msm.get_net_flux(source_states, sink_states)

    print("Gross flux (includes cycles):")
    print("  Non-zero entries:")
    for i in range(msm.n_states):
        for j in range(msm.n_states):
            if gross_flux[i, j] > 1e-10:
                print(f"    f[{i}→{j}] = {gross_flux[i, j]:.6e}")
    print()

    print("Net flux (removes cycles):")
    print("  Non-zero entries:")
    for i in range(msm.n_states):
        for j in range(msm.n_states):
            if net_flux[i, j] > 1e-10:
                print(f"    f+[{i}→{j}] = {net_flux[i, j]:.6e}")
    print()

    # 4. Pathway decomposition
    print("-" * 80)
    print("4. Pathway Decomposition")
    print("-" * 80)

    pathways, pathway_fluxes = msm.pathway_decomposition(
        source_states, sink_states, fraction=0.95, maxiter=1000
    )

    total_pathway_flux = np.sum(pathway_fluxes)
    print(f"Extracted {len(pathways)} pathways capturing 95% of flux")
    print(f"Total pathway flux: {total_pathway_flux:.6e}")
    print()

    for i, (path, path_flux) in enumerate(zip(pathways, pathway_fluxes)):
        fraction = path_flux / flux.total_flux if flux.total_flux > 0 else 0
        print(
            f"Pathway {i+1}: {' → '.join(map(str, path))} "
            f"(flux = {path_flux:.6e}, {fraction*100:.1f}%)"
        )
    print()

    # 5. Identify transition state ensemble
    print("-" * 80)
    print("5. Transition State Ensemble")
    print("-" * 80)

    ts_states = msm.identify_transition_state_ensemble(
        source_states, sink_states, tolerance=0.15
    )

    print(f"States with committor in [0.35, 0.65]:")
    for state in ts_states:
        print(
            f"  State {state}: q+ = {q_forward[state]:.4f}, "
            f"population = {pi[state]:.6f}"
        )
    print()

    # 6. Find bottleneck states
    print("-" * 80)
    print("6. Bottleneck States")
    print("-" * 80)

    bottlenecks = msm.find_bottleneck_states(source_states, sink_states, top_n=5)

    # Calculate flux through each state
    flux_through = 0.5 * (np.sum(gross_flux, axis=1) + np.sum(gross_flux, axis=0))

    print("Top 5 states by reactive flux:")
    for i, state in enumerate(bottlenecks):
        print(
            f"  {i+1}. State {state}: flux = {flux_through[state]:.6e}, "
            f"q+ = {q_forward[state]:.4f}"
        )
    print()

    # 7. Coarse-grain flux
    print("-" * 80)
    print("7. Coarse-Grained Flux")
    print("-" * 80)

    # Define macrostates: source, intermediate, sink
    intermediate_states = [i for i in range(10) if i not in source_states + sink_states]
    sets = [source_states, intermediate_states, sink_states]

    cg_sets, cg_flux = msm.coarse_grain_flux(source_states, sink_states, sets)

    print("Macrostates:")
    for i, s in enumerate(cg_sets):
        print(f"  Macrostate {i}: {sorted(s)}")
    print()

    print("Coarse-grained gross flux:")
    print(cg_flux.gross_flux)
    print()

    print("Coarse-grained net flux:")
    print(cg_flux.net_flux)
    print()

    print(f"Coarse-grained total flux: {cg_flux.total_flux:.6e}")
    print()

    # 8. Verify flux conservation
    print("-" * 80)
    print("8. Flux Conservation Check")
    print("-" * 80)

    print("Checking flux conservation at intermediate states...")
    for state in intermediate_states:
        flux_in = np.sum(net_flux[:, state])
        flux_out = np.sum(net_flux[state, :])
        error = abs(flux_in - flux_out)
        print(
            f"  State {state}: flux_in = {flux_in:.6e}, "
            f"flux_out = {flux_out:.6e}, error = {error:.6e}"
        )
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Source basin population: {np.sum([pi[i] for i in source_states]):.4f}")
    print(f"Sink basin population: {np.sum([pi[i] for i in sink_states]):.4f}")
    print(f"Reactive flux (A→B): {flux.total_flux:.6e}")
    print(f"Transition rate k_AB: {flux.rate:.6e}")
    print(f"Mean first passage time: {flux.mfpt:.2f} steps")
    print(f"Number of dominant pathways: {len(pathways)}")
    print(f"Transition state ensemble size: {len(ts_states)}")
    print()

    print("Analysis complete! Results saved to:", output_dir)


if __name__ == "__main__":
    main()
