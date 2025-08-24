from pathlib import Path

import numpy as np

from pmarlo.reporting.plots import save_fes_contour, save_transition_matrix_heatmap


def test_save_transition_matrix_heatmap(tmp_path):
    T = np.array([[0.5, 0.5], [0.5, 0.5]])
    path = save_transition_matrix_heatmap(T, output_dir=str(tmp_path))
    assert path is not None
    assert Path(path).exists()


def test_save_fes_contour(tmp_path):
    F = np.random.rand(5, 5)
    xedges = np.linspace(0, 1, 6)
    yedges = np.linspace(0, 1, 6)
    path = save_fes_contour(F, xedges, yedges, "X", "Y", str(tmp_path), "fes.png")
    assert path is not None
    assert Path(path).exists()
