"""Visualization utilities for Transition Path Theory analysis.

Provides visualization functions for TPT results including:
- Committor plots
- Flux network diagrams
- Pathway visualization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger("pmarlo.markov_state_model")


def plot_committor_distribution(
    q_forward: np.ndarray,
    q_backward: Optional[np.ndarray] = None,
    state_labels: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """Plot committor probability distribution.

    Args:
        q_forward: Forward committor probabilities
        q_backward: Backward committor probabilities (optional)
        state_labels: Labels for states (optional)
        ax: Matplotlib axes (optional, will create if not provided)
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_states = len(q_forward)
    states = np.arange(n_states)

    # Plot forward committor
    ax.plot(states, q_forward, "o-", label="Forward committor q+", linewidth=2)

    # Plot backward committor if provided
    if q_backward is not None:
        ax.plot(states, q_backward, "s-", label="Backward committor q-", linewidth=2)

    # Highlight transition state ensemble (q+ ≈ 0.5)
    ts_mask = (q_forward > 0.35) & (q_forward < 0.65)
    if np.any(ts_mask):
        ax.axhspan(0.35, 0.65, alpha=0.2, color="red", label="TSE region")

    ax.set_xlabel("State", fontsize=12)
    ax.set_ylabel("Committor Probability", fontsize=12)
    ax.set_title("Committor Probabilities", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    if state_labels:
        ax.set_xticks(states)
        ax.set_xticklabels(state_labels, rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_flux_network(
    flux_matrix: np.ndarray,
    threshold: float = 0.0,
    state_labels: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (12, 10),
    layout: str = "spring",
) -> Figure:
    """Plot flux network as a directed graph.

    Args:
        flux_matrix: Flux matrix (n_states x n_states)
        threshold: Minimum flux to display edge
        state_labels: Labels for nodes (optional)
        ax: Matplotlib axes (optional)
        figsize: Figure size
        layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')

    Returns:
        Matplotlib figure

    Raises:
        ImportError: If networkx not installed
    """
    import matplotlib.pyplot as plt

    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for flux network visualization. "
            "Install with: pip install networkx"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_states = flux_matrix.shape[0]

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes
    for i in range(n_states):
        label = state_labels[i] if state_labels else str(i)
        G.add_node(i, label=label)

    # Add edges with flux > threshold
    max_flux = np.max(flux_matrix)
    edges_added = []
    for i in range(n_states):
        for j in range(n_states):
            if flux_matrix[i, j] > threshold:
                # Edge width proportional to flux
                width = 0.5 + 5 * (flux_matrix[i, j] / max_flux)
                G.add_edge(i, j, weight=flux_matrix[i, j], width=width)
                edges_added.append((i, j))

    if len(edges_added) == 0:
        logger.warning(
            f"No edges above threshold {threshold}. "
            f"Max flux: {max_flux:.3e}. Consider lowering threshold."
        )
        return fig

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=1 / np.sqrt(n_states), iterations=50)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Draw nodes
    node_labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_nodes(
        G, pos, node_size=700, node_color="lightblue", ax=ax, alpha=0.9
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, ax=ax)

    # Draw edges with varying widths
    edges = G.edges()
    widths = [G[u][v]["width"] for u, v in edges]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        width=widths,
        alpha=0.6,
        edge_color="black",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
        ax=ax,
    )

    ax.set_title("Flux Network", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_pathways(
    pathways: List[List[int]],
    pathway_fluxes: np.ndarray,
    positions: Optional[np.ndarray] = None,
    state_labels: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (12, 8),
    top_n: int = 5,
) -> Figure:
    """Plot top reactive pathways.

    Args:
        pathways: List of pathways (each a list of state indices)
        pathway_fluxes: Flux through each pathway
        positions: State positions for plotting (n_states x 2)
        state_labels: Labels for states (optional)
        ax: Matplotlib axes (optional)
        figsize: Figure size
        top_n: Number of top pathways to plot

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Select top pathways
    top_indices = np.argsort(pathway_fluxes)[::-1][:top_n]
    top_pathways = [pathways[i] for i in top_indices]
    top_fluxes = pathway_fluxes[top_indices]

    # If no positions provided, create simple linear positions
    if positions is None:
        all_states = set()
        for path in top_pathways:
            all_states.update(path)
        n_states = max(all_states) + 1
        positions = np.zeros((n_states, 2))
        positions[:, 0] = np.arange(n_states)
        positions[:, 1] = 0

    # Normalize fluxes for plotting
    max_flux = np.max(top_fluxes)

    # Plot each pathway
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(top_pathways)))

    for i, (path, flux) in enumerate(zip(top_pathways, top_fluxes)):
        # Line width proportional to flux
        linewidth = 1 + 5 * (flux / max_flux)

        # Plot pathway
        path_positions = positions[path]
        ax.plot(
            path_positions[:, 0],
            path_positions[:, 1],
            "o-",
            color=colors[i],
            linewidth=linewidth,
            markersize=8,
            alpha=0.7,
            label=f"Path {i+1}: flux={flux:.2e}",
        )

    # Plot all states
    all_states = set()
    for path in top_pathways:
        all_states.update(path)

    for state in all_states:
        label = state_labels[state] if state_labels else str(state)
        ax.annotate(
            label,
            xy=positions[state],
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Position", fontsize=12)
    ax.set_title(f"Top {len(top_pathways)} Reactive Pathways", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_tpt_summary(
    q_forward: np.ndarray,
    flux_matrix: np.ndarray,
    pathways: List[List[int]],
    pathway_fluxes: np.ndarray,
    output_path: Optional[str | Path] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> Figure:
    """Create a comprehensive TPT summary plot.

    Args:
        q_forward: Forward committor probabilities
        flux_matrix: Net flux matrix
        pathways: List of pathways
        pathway_fluxes: Flux through each pathway
        output_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. Committor distribution
    ax1 = fig.add_subplot(gs[0, 0])
    plot_committor_distribution(q_forward, ax=ax1)

    # 2. Flux network
    ax2 = fig.add_subplot(gs[0, 1])
    threshold = np.max(flux_matrix) * 0.01  # Show edges with >1% of max flux
    try:
        plot_flux_network(flux_matrix, threshold=threshold, ax=ax2, layout="circular")
    except ImportError:
        ax2.text(
            0.5,
            0.5,
            "networkx required for flux network\nInstall: pip install networkx",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.axis("off")

    # 3. Pathway flux distribution
    ax3 = fig.add_subplot(gs[1, 0])
    top_n = min(10, len(pathways))
    top_indices = np.argsort(pathway_fluxes)[::-1][:top_n]
    top_fluxes = pathway_fluxes[top_indices]
    ax3.bar(range(len(top_fluxes)), top_fluxes, color="steelblue")
    ax3.set_xlabel("Pathway Rank", fontsize=12)
    ax3.set_ylabel("Flux", fontsize=12)
    ax3.set_title("Top Pathway Fluxes", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Flux through states
    ax4 = fig.add_subplot(gs[1, 1])
    flux_through_state = 0.5 * (np.sum(flux_matrix, axis=1) + np.sum(flux_matrix, axis=0))
    states = np.arange(len(flux_through_state))
    ax4.bar(states, flux_through_state, color="coral")
    ax4.set_xlabel("State", fontsize=12)
    ax4.set_ylabel("Total Flux", fontsize=12)
    ax4.set_title("Flux Through States", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle("TPT Analysis Summary", fontsize=16, fontweight="bold", y=0.98)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved TPT summary plot to {output_path}")

    return fig


__all__ = [
    "plot_committor_distribution",
    "plot_flux_network",
    "plot_pathways",
    "plot_tpt_summary",
]

