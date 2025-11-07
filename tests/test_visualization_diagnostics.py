import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pmarlo.visualization.diagnostics import (
    create_fes_validation_plot,
    create_sampling_validation_plot,
)


def test_create_sampling_validation_plot_basic():
    projection = [
        np.linspace(-1.0, 1.0, 200).reshape(-1, 1),
        np.sin(np.linspace(0.0, 2 * np.pi, 150)).reshape(-1, 1),
    ]

    fig = create_sampling_validation_plot(
        projection_data=projection,
        run_labels=["run-a", "run-b"],
        max_length=120,
        hist_bins=50,
        stride=5,
    )
    assert isinstance(fig, Figure)
    assert fig.axes, "Expected at least one axis on the sampling plot"
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
        dtraj_data=dtraj,
        cluster_centers=cluster_centers,
        max_length=90,
        stride=4,
    )
    assert isinstance(fig, Figure)
    assert fig.axes, "Expected at least one axis for discrete overlay plot"
    plt.close(fig)


def test_create_sampling_validation_plot_requires_projection():
    with pytest.raises(ValueError, match="projection_data must contain at least one trajectory"):
        create_sampling_validation_plot([])


def test_create_sampling_validation_plot_ignores_empty_inputs():
    projection = [
        np.empty((0, 2)),
        np.linspace(-1.0, 1.0, 50).reshape(-1, 1),
    ]

    fig = create_sampling_validation_plot(projection_data=projection)
    assert isinstance(fig, Figure)
    plt.close(fig)


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

    with pytest.raises(ValueError, match="Coordinate arrays xx and yy must have identical shapes"):
        create_fes_validation_plot(fes_grid=(xx, yy), fes_data=fes)

