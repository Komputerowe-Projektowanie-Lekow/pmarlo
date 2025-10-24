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
    figsize: Tuple[int, int] = (15, 6),
    titles: List[str] = ["Forward Committor", "Backward Committor"],
    cmap: str = "coolwarm",
    xlabel_2d: str = "x coordinate",
    ylabel_2d: str = "y coordinate",
    xlabel_1d: str = "State index",
    ylabel_1d: str = "Committor probability",
    bar_color: str = "C0",
) -> plt.Figure:
    """Plot forward and backward committors.

    Args:
        forward_committor: Forward committor values.
        backward_committor: Backward committor values.
        grid_size: Optional grid size for 2D visualization.
        output_path: Optional path to save figure.
        figsize: Figure size for the committor plots.
        titles: Titles for the forward and backward committor plots.
        cmap: Colormap used for the 2D committor visualization.
        xlabel_2d: Label for the x-axis of the 2D plot.
        ylabel_2d: Label for the y-axis of the 2D plot.
        xlabel_1d: Label for the x-axis of the 1D plot.
        ylabel_1d: Label for the y-axis of the 1D plot.
        bar_color: Color of the bars in the 1D committor plot.

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Ensure both committor arrays have the same length so every state is shown.
    n_states = int(max(len(forward_committor), len(backward_committor)))
    forward_values = np.zeros(n_states, dtype=float)
    backward_values = np.zeros(n_states, dtype=float)
    forward_values[: len(forward_committor)] = forward_committor
    backward_values[: len(backward_committor)] = backward_committor

    committors = [forward_values, backward_values]

    for i, ax in enumerate(axes):
        ax.set_title(titles[i])

        if grid_size is not None:
            # 2D visualization
            Q = committors[i].reshape(grid_size)
            im = ax.imshow(Q, interpolation="nearest", origin="lower", cmap=cmap)
            plt.colorbar(im, ax=ax)
            ax.set_xlabel(xlabel_2d)
            ax.set_ylabel(ylabel_2d)
        else:
            # 1D bar plot covering every state index
            state_indices = np.arange(n_states)
            ax.bar(state_indices, committors[i], color=bar_color)
            ax.set_xticks(state_indices)
            ax.set_xlim(-0.5, n_states - 0.5)
            ax.set_xlabel(xlabel_1d)
            ax.set_ylabel(ylabel_1d)

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
    figsize: Tuple[int, int] = (16, 8),
    titles: List[str] = ["Gross Flux", "Net Flux"],
    node_color: str = "gray",
    source_color: str = "green",
    sink_color: str = "red",
    node_size: int = 200,
    node_zorder: int = 10,
    node_edgecolor: str = "black",
    lw_scale: float = 1000.0,
    lw_factor: float = 0.5,
    arrow_alpha: float = 0.6,
    positive_flux_color: str = "red",
    negative_flux_color: str = "blue",
    label_offset_factor: float = 1.15,
    axis_limit_padding: float = 1.5,
) -> plt.Figure:
    """Plot gross and net flux networks.

    Args:
        flux_matrix: Gross flux matrix.
        net_flux: Net flux matrix.
        source_states: Optional source state indices.
        sink_states: Optional sink state indices.
        output_path: Optional path to save figure.
        connection_threshold: Minimum flux to show connection.
        figsize: Figure size for the gross and net flux plots.
        titles: Titles for the gross and net flux plots.
        node_color: Default color for intermediate nodes.
        source_color: Color for source nodes.
        sink_color: Color for sink nodes.
        node_size: Size of the nodes in the scatter plot.
        node_zorder: Z-order for nodes to control layering.
        node_edgecolor: Edge color for nodes.
        lw_scale: Scale factor applied before the logarithm when computing arrow line widths.
        lw_factor: Multiplicative factor applied to the logarithmic arrow line width.
        arrow_alpha: Transparency for arrows.
        positive_flux_color: Color used when the flux value is positive.
        negative_flux_color: Color used when the flux value is negative or zero.
        label_offset_factor: Radial factor for positioning node labels.
        axis_limit_padding: Extent of the axis limits for both dimensions.

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
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
                    linewidth = np.log1p(F[i_state, j_state] * lw_scale) * lw_factor
                    ax.annotate(
                        "",
                        xy=(pos_x[j_state], pos_y[j_state]),
                        xytext=(pos_x[i_state], pos_y[i_state]),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=linewidth,
                            alpha=arrow_alpha,
                            color=(
                                positive_flux_color
                                if F[i_state, j_state] > 0
                                else negative_flux_color
                            ),
                        ),
                    )

        # Plot nodes
        node_colors = [node_color] * n_states
        if source_states:
            for s in source_states:
                if s < n_states:
                    node_colors[s] = source_color
        if sink_states:
            for s in sink_states:
                if s < n_states:
                    node_colors[s] = sink_color

        ax.scatter(
            pos_x,
            pos_y,
            c=node_colors,
            s=node_size,
            zorder=node_zorder,
            edgecolors=node_edgecolor,
        )

        # Add state labels
        for state in range(n_states):
            ax.text(
                pos_x[state] * label_offset_factor,
                pos_y[state] * label_offset_factor,
                str(state),
                ha="center",
                va="center",
            )

        ax.set_xlim(-axis_limit_padding, axis_limit_padding)
        ax.set_ylim(-axis_limit_padding, axis_limit_padding)
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
    figsize: Tuple[int, int] = (12, 8),
    title: str = "Top {n_paths} Pathways by Capacity",
    bar_alpha: float = 0.7,
    bar_color: str = "C0",
    path_separator: str = " → ",
    label_truncation: int = 50,
    capacity_label_offset: float = 0.01,
    xlabel: str = "Flux Fraction",
    ylabel: str = "Pathway",
    path_label_template: str = "Path {index}",
    xlim_max_padding_factor: float = 1.2,
) -> plt.Figure:
    """Plot pathway decomposition.

    Args:
        pathways: List of pathways (state sequences).
        pathway_fluxes: Flux capacity of each pathway.
        output_path: Optional path to save figure.
        max_paths: Maximum number of pathways to display.
        figsize: Figure size for the pathway bar chart.
        title: Title for the pathway plot; receives ``n_paths`` via ``str.format``.
        bar_alpha: Transparency applied to the pathway bars.
        bar_color: Color applied to the pathway bars.
        path_separator: Separator used when joining pathway state labels.
        label_truncation: Maximum number of characters from the pathway label to display before appending ellipsis.
        capacity_label_offset: Horizontal offset for the capacity annotation.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        path_label_template: Template used when constructing y-axis tick labels.
        xlim_max_padding_factor: Factor applied to the maximum fraction to set the x-axis limit.

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_paths = min(len(pathways), max_paths)
    if n_paths == 0:
        return fig

    total_flux = float(np.sum(pathway_fluxes[:n_paths]))
    if total_flux == 0:
        return fig

    y_positions = np.arange(n_paths)

    for i in range(n_paths):
        path = pathways[i]
        capacity = pathway_fluxes[i]
        fraction = capacity / total_flux

        # Draw pathway
        path_str = path_separator.join(map(str, path))
        truncated_path = (
            path_str
            if len(path_str) <= label_truncation
            else f"{path_str[:label_truncation]}..."
        )
        ax.barh(
            y_positions[i],
            fraction,
            label=f"{path_label_template.format(index=i+1)}: {truncated_path}",
            alpha=bar_alpha,
            color=bar_color,
        )

        # Add capacity label
        ax.text(
            fraction + capacity_label_offset,
            y_positions[i],
            f"{capacity:.2e} ({fraction*100:.1f}%)",
            va="center",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title.format(n_paths=n_paths))
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [path_label_template.format(index=i + 1) for i in range(n_paths)]
    )
    max_fraction = float(np.max(pathway_fluxes[:n_paths] / total_flux))
    xmax = max(capacity_label_offset, max_fraction * xlim_max_padding_factor)
    ax.set_xlim(0, xmax)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_coarse_grained_flux(
    cg_gross_flux: np.ndarray,
    cg_net_flux: np.ndarray,
    set_labels: List[str],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 7),
    titles: List[str] = ["Coarse-Grained Gross Flux", "Coarse-Grained Net Flux"],
    node_size: int = 1500,
    node_color: str = "lightblue",
    arrowstyle: str = "-|>",
    arrowsize: int = 20,
    connectionstyle_rad: float = 0.2,
    edge_width: float = 2.0,
    edge_color: str = "black",
) -> plt.Figure:
    """Plot coarse-grained flux network.

    Args:
        cg_gross_flux: Coarse-grained gross flux.
        cg_net_flux: Coarse-grained net flux.
        set_labels: Labels for each coarse-grained set.
        output_path: Optional path to save figure.
        figsize: Figure size for the coarse-grained flux plots.
        titles: Titles for the gross and net coarse-grained flux plots.
        node_size: Node size for the coarse-grained network visualization.
        node_color: Node color for the coarse-grained network visualization.
        arrowstyle: Matplotlib arrow style for network edges.
        arrowsize: Arrow size for network edges.
        connectionstyle_rad: Arc radius for curved edges.
        edge_width: Width of the edges in the network plot.
        edge_color: Color of the edges in the network plot.

    Returns:
        Matplotlib figure
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("Coarse-grained flux visualization requires networkx")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    fluxes = [cg_gross_flux, cg_net_flux]

    n_sets = cg_gross_flux.shape[0]

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
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=node_size, node_color=node_color
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax, labels=nx.get_node_attributes(G, "title")
        )

        edge_labels = nx.get_edge_attributes(G, "title")
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels)

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            arrowstyle=arrowstyle,
            arrowsize=arrowsize,
            connectionstyle=f"arc3, rad={connectionstyle_rad}",
            width=edge_width,
            edge_color=edge_color,
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
    committors_figsize: Tuple[int, int] = (15, 6),
    committors_titles: List[str] = ["Forward Committor", "Backward Committor"],
    committors_cmap: str = "coolwarm",
    committors_xlabel_2d: str = "x coordinate",
    committors_ylabel_2d: str = "y coordinate",
    committors_xlabel_1d: str = "State index",
    committors_ylabel_1d: str = "Committor probability",
    committors_bar_color: str = "C0",
    flux_figsize: Tuple[int, int] = (16, 8),
    flux_titles: List[str] = ["Gross Flux", "Net Flux"],
    flux_node_color: str = "gray",
    flux_source_color: str = "green",
    flux_sink_color: str = "red",
    flux_node_size: int = 200,
    flux_node_zorder: int = 10,
    flux_node_edgecolor: str = "black",
    flux_lw_scale: float = 1000.0,
    flux_lw_factor: float = 0.5,
    flux_arrow_alpha: float = 0.6,
    flux_positive_flux_color: str = "red",
    flux_negative_flux_color: str = "blue",
    flux_label_offset_factor: float = 1.15,
    flux_axis_limit_padding: float = 1.5,
    flux_connection_threshold: float = 0.0,
    pathways_max_paths: int = 10,
    pathways_figsize: Tuple[int, int] = (12, 8),
    pathways_title: str = "Top {n_paths} Pathways by Capacity",
    pathways_bar_alpha: float = 0.7,
    pathways_bar_color: str = "C0",
    pathways_separator: str = " → ",
    pathways_label_truncation: int = 50,
    pathways_capacity_label_offset: float = 0.01,
    pathways_xlabel: str = "Flux Fraction",
    pathways_ylabel: str = "Pathway",
    pathways_label_template: str = "Path {index}",
    pathways_xlim_max_padding_factor: float = 1.2,
    committors_filename: str = "committors.png",
    flux_network_filename: str = "flux_network.png",
    pathways_filename: str = "pathways.png",
) -> None:
    """Create all TPT visualization plots.

    Args:
        tpt_result: TPTResult object.
        output_dir: Directory to save plots.
        grid_size: Optional grid size for 2D plots.
        committors_figsize: Figure size for committor plots.
        committors_titles: Titles for the committor plots.
        committors_cmap: Colormap for the committor heatmap.
        committors_xlabel_2d: X-axis label for the committor heatmap.
        committors_ylabel_2d: Y-axis label for the committor heatmap.
        committors_xlabel_1d: X-axis label for the committor bar plot.
        committors_ylabel_1d: Y-axis label for the committor bar plot.
        committors_bar_color: Bar color for the committor bar plot.
        flux_figsize: Figure size for the flux network plots.
        flux_titles: Titles for the flux network plots.
        flux_node_color: Default node color for the flux network plot.
        flux_source_color: Color for source nodes in the flux network plot.
        flux_sink_color: Color for sink nodes in the flux network plot.
        flux_node_size: Size for nodes in the flux network plot.
        flux_node_zorder: Z-order for nodes in the flux network plot.
        flux_node_edgecolor: Edge color for nodes in the flux network plot.
        flux_lw_scale: Scale factor before the logarithm in the flux network arrow width.
        flux_lw_factor: Multiplicative factor for the flux network arrow width.
        flux_arrow_alpha: Transparency for arrows in the flux network plot.
        flux_positive_flux_color: Color for positive flux arrows.
        flux_negative_flux_color: Color for non-positive flux arrows.
        flux_label_offset_factor: Factor controlling node label placement radius.
        flux_axis_limit_padding: Axis extent for the flux network plots.
        flux_connection_threshold: Minimum flux required to draw an edge.
        pathways_max_paths: Maximum number of pathways to plot.
        pathways_figsize: Figure size for pathway plots.
        pathways_title: Title for the pathway plot.
        pathways_bar_alpha: Bar transparency for pathways.
        pathways_bar_color: Bar color for pathways.
        pathways_separator: Separator used to join pathway state labels.
        pathways_label_truncation: Maximum characters displayed for a pathway label.
        pathways_capacity_label_offset: Horizontal offset for capacity annotations.
        pathways_xlabel: X-axis label for pathway plots.
        pathways_ylabel: Y-axis label for pathway plots.
        pathways_label_template: Template used for pathway labels.
        pathways_xlim_max_padding_factor: Factor applied to determine pathway plot x-axis limit.
        committors_filename: Filename for the committor plot output.
        flux_network_filename: Filename for the flux network plot output.
        pathways_filename: Filename for the pathways plot output.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Committors
    plot_committors(
        tpt_result.forward_committor,
        tpt_result.backward_committor,
        grid_size=grid_size,
        output_path=str(output_path / committors_filename),
        figsize=committors_figsize,
        titles=committors_titles,
        cmap=committors_cmap,
        xlabel_2d=committors_xlabel_2d,
        ylabel_2d=committors_ylabel_2d,
        xlabel_1d=committors_xlabel_1d,
        ylabel_1d=committors_ylabel_1d,
        bar_color=committors_bar_color,
    )

    # Flux networks
    plot_flux_network(
        tpt_result.flux_matrix,
        tpt_result.net_flux,
        source_states=tpt_result.source_states.tolist(),
        sink_states=tpt_result.sink_states.tolist(),
        output_path=str(output_path / flux_network_filename),
        connection_threshold=flux_connection_threshold,
        figsize=flux_figsize,
        titles=flux_titles,
        node_color=flux_node_color,
        source_color=flux_source_color,
        sink_color=flux_sink_color,
        node_size=flux_node_size,
        node_zorder=flux_node_zorder,
        node_edgecolor=flux_node_edgecolor,
        lw_scale=flux_lw_scale,
        lw_factor=flux_lw_factor,
        arrow_alpha=flux_arrow_alpha,
        positive_flux_color=flux_positive_flux_color,
        negative_flux_color=flux_negative_flux_color,
        label_offset_factor=flux_label_offset_factor,
        axis_limit_padding=flux_axis_limit_padding,
    )

    # Pathways
    if len(tpt_result.pathways) > 0:
        plot_pathways(
            tpt_result.pathways,
            tpt_result.pathway_fluxes,
            output_path=str(output_path / pathways_filename),
            max_paths=pathways_max_paths,
            figsize=pathways_figsize,
            title=pathways_title,
            bar_alpha=pathways_bar_alpha,
            bar_color=pathways_bar_color,
            path_separator=pathways_separator,
            label_truncation=pathways_label_truncation,
            capacity_label_offset=pathways_capacity_label_offset,
            xlabel=pathways_xlabel,
            ylabel=pathways_ylabel,
            path_label_template=pathways_label_template,
            xlim_max_padding_factor=pathways_xlim_max_padding_factor,
        )

    plt.close("all")  # Clean up

