import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pmarlo.visualization.diagnostics import (
    create_fes_validation_plot,
    create_sampling_validation_plot,
    create_shard_frame_histogram,
)


def test_create_sampling_validation_plot_basic():
    projection = [
        np.linspace(-1.0, 1.0, 200).reshape(-1, 1),
        np.sin(np.linspace(0.0, 2 * np.pi, 150)).reshape(-1, 1),
    ]

    fig = create_sampling_validation_plot(
        projection_data=projection,
        run_labels=["run-a", "run-b"],
        metabiased_runs=[False, True],
        max_length=120,
        hist_bins=50,
        stride=5,
    )
    assert isinstance(fig, Figure)
    assert fig.axes, "Expected at least one axis on the sampling plot"

    ax = fig.axes[0]
    lines = ax.lines
    assert len(lines) == 2
    assert lines[0].get_linestyle() == "-"
    assert lines[1].get_linestyle() == "--"

    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert any("run-b" in label and "metabiased" in label for label in legend_labels)
    assert "Standard Run" in legend_labels
    assert "Metabiased Run" in legend_labels
    legend_title = legend.get_title().get_text()
    assert "Runs:" in legend_title
    assert "standard" in legend_title
    assert "metabiased" in legend_title
    plt.close(fig)


def test_create_sampling_validation_plot_with_discrete_overlay():
    projection = [
        np.column_stack(
            (
                np.linspace(-1.0, 1.0, 100),
                np.linspace(0.0, 2.0, 100),
            )
        ),
        np.column_stack(
            (
                np.cos(np.linspace(0.0, 2 * np.pi, 80)),
                np.sin(np.linspace(0.0, 2 * np.pi, 80)),
            )
        ),
    ]
    cluster_centers = np.array(
        [
            [0.0, 0.0],
            [0.5, 1.0],
            [-0.5, -1.0],
        ]
    )
    dtraj = [
        np.array([0, 1, 1, 2, 0, 2, 1, 0]),
        np.array([2, 2, 1, 0, 1, 2, 0, 0]),
    ]

    fig = create_sampling_validation_plot(
        projection_data=projection,
        run_labels=["run-a", "run-b"],
        metabiased_runs=[False, True],
        dtraj_data=dtraj,
        cluster_centers=cluster_centers,
        max_length=90,
        stride=4,
    )
    assert isinstance(fig, Figure)
    assert fig.axes, "Expected at least one axis for discrete overlay plot"

    overlay_lines = fig.axes[0].lines[2:]
    assert overlay_lines, "Expected discrete overlay lines"
    assert all(line.get_linestyle() == ":" for line in overlay_lines)
    plt.close(fig)


def test_create_sampling_validation_plot_requires_projection():
    with pytest.raises(
        ValueError, match="projection_data must contain at least one trajectory"
    ):
        create_sampling_validation_plot([])


def test_create_sampling_validation_plot_ignores_empty_inputs():
    projection = [
        np.empty((0, 2)),
        np.linspace(-1.0, 1.0, 50).reshape(-1, 1),
    ]

    fig = create_sampling_validation_plot(projection_data=projection)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_create_sampling_validation_plot_rejects_metabiased_length_mismatch():
    projection = [
        np.linspace(-1.0, 1.0, 20).reshape(-1, 1),
        np.linspace(-1.0, 1.0, 25).reshape(-1, 1),
    ]

    with pytest.raises(
        ValueError, match="metabiased_runs must have the same length as projection_data"
    ):
        create_sampling_validation_plot(
            projection_data=projection,
            metabiased_runs=[True],
        )


def test_create_fes_validation_plot_basic():
    x = y = np.linspace(-1.0, 1.0, 32)
    xx, yy = np.meshgrid(x, y)
    fes = xx**2 + yy**2

    fig = create_fes_validation_plot(
        fes_grid=(xx, yy),
        fes_data=fes,
        max_kt=5.0,
        levels=15,
        cmap="plasma",
    )
    assert isinstance(fig, Figure)
    assert fig.axes, "Expected at least one axis on the FES plot"
    plt.close(fig)


def test_create_fes_validation_plot_rejects_bad_grid():
    xx = np.zeros((4, 4))
    yy = np.zeros((5, 5))
    fes = np.zeros((4, 4))

    with pytest.raises(
        ValueError, match="Coordinate arrays xx and yy must have identical shapes"
    ):
        create_fes_validation_plot(fes_grid=(xx, yy), fes_data=fes)


def test_create_shard_frame_histogram_basic():
    frame_counts = [842, 913, 760, 1012]
    shard_labels = ["shard-a", "shard-b", "shard-c", "shard-d"]

    fig = create_shard_frame_histogram(
        frame_counts,
        shard_labels=shard_labels,
        max_label_count=10,
        figsize=(6.0, 4.0),
    )
    assert isinstance(fig, Figure)
    assert fig.axes, "Expected histogram axes"
    ax = fig.axes[0]
    assert ax.get_title() == "Frames per shard"
    assert ax.get_ylabel() == "Frames"
    assert [tick.get_text() for tick in ax.get_xticklabels()] == shard_labels
    assert [patch.get_height() for patch in ax.patches] == frame_counts
    plt.close(fig)


def test_create_shard_frame_histogram_limits_labels():
    frame_counts = list(range(1, 9))
    labels = [f"shard-{idx}" for idx in range(len(frame_counts))]

    fig = create_shard_frame_histogram(
        frame_counts,
        shard_labels=labels,
        max_label_count=4,
    )
    ax = fig.axes[0]
    xtick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert xtick_labels[0] == "1"
    assert xtick_labels[-1] == str(len(frame_counts))
    plt.close(fig)


def test_create_shard_frame_histogram_validates_inputs():
    with pytest.raises(ValueError, match="frame_counts must contain at least one value"):
        create_shard_frame_histogram([])

    with pytest.raises(ValueError, match="frame_counts must contain only finite values"):
        create_shard_frame_histogram([1.0, np.nan])

    with pytest.raises(ValueError, match="frame_counts must be non-negative"):
        create_shard_frame_histogram([-1])

    with pytest.raises(ValueError, match="shard_labels length must match"):
        create_shard_frame_histogram([1, 2], shard_labels=["only-one"])
