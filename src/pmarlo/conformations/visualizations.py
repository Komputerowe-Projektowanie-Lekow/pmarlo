"""Visualization functions for conformations analysis.

Based on deeptime TPT examples:
https://deeptime-ml.github.io/latest/notebooks/tpt.html
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_committors(
    forward_committor: np.ndarray,
    backward_committor: np.ndarray,
    grid_size: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot forward and backward committors.

    Args:
        forward_committor: Forward committor values
        backward_committor: Backward committor values
        grid_size: Optional grid size for 2D visualization
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    titles = ["Forward Committor", "Backward Committor"]
    committors = [forward_committor, backward_committor]

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])

        if grid_size is not None:
            # 2D visualization
            Q = committors[i].reshape(grid_size)
            im = ax.imshow(Q, interpolation="nearest", origin="lower", cmap="coolwarm")
            plt.colorbar(im, ax=ax)
            ax.set_xlabel("x coordinate")
            ax.set_ylabel("y coordinate")
        else:
            # 1D bar plot
            ax.bar(range(len(committors[i])), committors[i])
            ax.set_xlabel("State index")
            ax.set_ylabel("Committor probability")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_flux_network(
    flux_matrix: np.ndarray,
    net_flux: np.ndarray,
    source_states: Optional[List[int]] = None,
    sink_states: Optional[List[int]] = None,
    output_path: Optional[str] = None,
    connection_threshold: float = 0.0,
) -> plt.Figure:
    """Plot gross and net flux networks.

    Args:
        flux_matrix: Gross flux matrix
        net_flux: Net flux matrix  
        source_states: Optional source state indices
        sink_states: Optional sink state indices
        output_path: Optional path to save figure
        connection_threshold: Minimum flux to show connection

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    titles = ["Gross Flux", "Net Flux"]
    fluxes = [flux_matrix, net_flux]

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])

        F = fluxes[i]
        n_states = F.shape[0]

        # Create network layout (circular)
        theta = np.linspace(0, 2 * np.pi, n_states, endpoint=False)
        pos_x = np.cos(theta)
        pos_y = np.sin(theta)

        # Plot connections
        for i_state in range(n_states):
            for j_state in range(n_states):
                if F[i_state, j_state] > connection_threshold:
                    # Draw arrow
                    ax.annotate(
                        "",
                        xy=(pos_x[j_state], pos_y[j_state]),
                        xytext=(pos_x[i_state], pos_y[i_state]),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=np.log1p(F[i_state, j_state] * 1000) * 0.5,
                            alpha=0.6,
                            color="red" if F[i_state, j_state] > 0 else "blue",
                        ),
                    )

        # Plot nodes
        node_colors = ["gray"] * n_states
        if source_states:
            for s in source_states:
                if s < n_states:
                    node_colors[s] = "green"
        if sink_states:
            for s in sink_states:
                if s < n_states:
                    node_colors[s] = "red"

        ax.scatter(pos_x, pos_y, c=node_colors, s=200, zorder=10, edgecolors="black")

        # Add state labels
        for state in range(n_states):
            ax.text(
                pos_x[state] * 1.15,
                pos_y[state] * 1.15,
                str(state),
                ha="center",
                va="center",
            )

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_pathways(
    pathways: List[List[int]],
    pathway_fluxes: np.ndarray,
    output_path: Optional[str] = None,
    max_paths: int = 10,
) -> plt.Figure:
    """Plot pathway decomposition.

    Args:
        pathways: List of pathways (state sequences)
        pathway_fluxes: Flux capacity of each pathway
        output_path: Optional path to save figure
        max_paths: Maximum number of pathways to display

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    n_paths = min(len(pathways), max_paths)
    total_flux = np.sum(pathway_fluxes) if len(pathway_fluxes) > 0 else 1.0

    y_positions = np.arange(n_paths)

    for i in range(n_paths):
        path = pathways[i]
        capacity = pathway_fluxes[i]
        fraction = capacity / total_flux if total_flux > 0 else 0

        # Draw pathway
        path_str = " → ".join(map(str, path))
        ax.barh(
            y_positions[i],
            fraction,
            label=f"Path {i+1}: {path_str[:50]}...",
            alpha=0.7,
        )

        # Add capacity label
        ax.text(
            fraction + 0.01,
            y_positions[i],
            f"{capacity:.2e} ({fraction*100:.1f}%)",
            va="center",
        )

    ax.set_xlabel("Flux Fraction")
    ax.set_ylabel("Pathway")
    ax.set_title(f"Top {n_paths} Pathways by Capacity")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"Path {i+1}" for i in range(n_paths)])
    ax.set_xlim(0, max(1.0, max(pathway_fluxes) / total_flux * 1.2))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_coarse_grained_flux(
    cg_gross_flux: np.ndarray,
    cg_net_flux: np.ndarray,
    set_labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot coarse-grained flux network.

    Args:
        cg_gross_flux: Coarse-grained gross flux
        cg_net_flux: Coarse-grained net flux
        set_labels: Optional labels for each set
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("Coarse-grained flux visualization requires networkx")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    titles = ["Coarse-Grained Gross Flux", "Coarse-Grained Net Flux"]
    fluxes = [cg_gross_flux, cg_net_flux]

    n_sets = cg_gross_flux.shape[0]
    if set_labels is None:
        set_labels = [f"Set {i+1}" for i in range(n_sets)]

    for idx, ax in enumerate(axes):
        ax.set_title(titles[idx])

        F = fluxes[idx]
        G = nx.DiGraph()

        # Add nodes
        for i in range(n_sets):
            G.add_node(i, title=set_labels[i])

        # Add edges
        for i in range(n_sets):
            for j in range(n_sets):
                if F[i, j] > 0:
                    G.add_edge(i, j, weight=F[i, j], title=f"{F[i, j]:.3e}")

        # Layout
        pos = nx.circular_layout(G)

        # Draw
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1500, node_color="lightblue")
        nx.draw_networkx_labels(
            G, pos, ax=ax, labels=nx.get_node_attributes(G, "title")
        )

        edge_labels = nx.get_edge_attributes(G, "title")
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrowstyle="-|>",
            arrowsize=20,
            connectionstyle="arc3, rad=0.2",
            width=2,
        )

        ax.axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_tpt_summary(
    tpt_result: Any,
    output_dir: str,
    grid_size: Optional[Tuple[int, int]] = None,
) -> None:
    """Create all TPT visualization plots.

    Args:
        tpt_result: TPTResult object
        output_dir: Directory to save plots
        grid_size: Optional grid size for 2D plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Committors
    plot_committors(
        tpt_result.forward_committor,
        tpt_result.backward_committor,
        grid_size=grid_size,
        output_path=str(output_path / "committors.png"),
    )

    # Flux networks
    plot_flux_network(
        tpt_result.flux_matrix,
        tpt_result.net_flux,
        source_states=tpt_result.source_states.tolist(),
        sink_states=tpt_result.sink_states.tolist(),
        output_path=str(output_path / "flux_network.png"),
    )

    # Pathways
    if len(tpt_result.pathways) > 0:
        plot_pathways(
            tpt_result.pathways,
            tpt_result.pathway_fluxes,
            output_path=str(output_path / "pathways.png"),
        )

    plt.close("all")  # Clean up

