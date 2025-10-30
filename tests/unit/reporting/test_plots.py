from pathlib import Path

import numpy as np
import pytest

from pmarlo.reporting.plots import (
    save_fes_contour,
    save_transition_matrix_heatmap,
    plot_sampling_validation,
)


def test_save_transition_matrix_heatmap(tmp_path):
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    path = save_transition_matrix_heatmap(T, output_dir=str(tmp_path))
    assert path is not None
    assert Path(path).exists()


def test_save_fes_contour(tmp_path):
    F = np.linspace(0.0, 1.0, 25, dtype=float).reshape(5, 5)
    xedges = np.linspace(0, 1, 6)
    yedges = np.linspace(0, 1, 6)
    path = save_fes_contour(F, xedges, yedges, "X", "Y", str(tmp_path), "fes.png")
    assert path is not None
    assert Path(path).exists()


def test_save_fes_contour_rejects_sparse_surface(tmp_path):
    F = np.full((5, 5), np.inf)
    F[0, 0] = 0.0
    xedges = np.linspace(0, 1, 6)
    yedges = np.linspace(0, 1, 6)
    with pytest.raises(ValueError, match="too sparse"):
        save_fes_contour(F, xedges, yedges, "X", "Y", str(tmp_path), "fes.png")


def test_plot_sampling_validation_rejects_missing_data():
    with pytest.raises(ValueError, match="at least one trajectory"):
        plot_sampling_validation([])


def test_plot_sampling_validation_rejects_empty_trajectory():
    with pytest.raises(ValueError, match="contains empty trajectories"):
        plot_sampling_validation([np.array([1.0, 2.0]), np.array([])])


def test_plot_sampling_validation_rejects_unknown_colormap():
    data = [np.linspace(0.0, 1.0, 10)]
    with pytest.raises(ValueError, match="Unknown colormap"):
        plot_sampling_validation(data, cmap_name="this_cmap_does_not_exist")
