from __future__ import annotations

from typing import cast

import numpy as np


def pca_reduce(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Simple PCA using SVD with mean-centering.

    Returns projected data (n_frames, n_components).
    """
    if X.size == 0:
        return np.zeros((X.shape[0], n_components), dtype=float)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    # Economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(max(1, min(n_components, Vt.shape[0])))
    return cast(np.ndarray, U[:, :k] @ np.diag(S[:k]))


def tica_reduce(X: np.ndarray, lag: int = 10, n_components: int = 2) -> np.ndarray:
    """Lightweight TICA: solve C0^{-1} C_tau eigenproblem and project.

    - No whitening; returns projected coordinates.
    - Assumes X is (n_frames, n_features)
    """
    if X.size == 0:
        return np.zeros((X.shape[0], n_components), dtype=float)
    Xc = X - np.mean(X, axis=0, keepdims=True)
    if Xc.shape[0] <= lag + 1:
        return pca_reduce(Xc, n_components=n_components)
    X0 = Xc[:-lag]
    X1 = Xc[lag:]
    C0 = X0.T @ X0
    Ctau = X0.T @ X1
    # Regularize C0 for stability
    eps = 1e-6
    C0 += eps * np.eye(C0.shape[0])
    A = np.linalg.solve(C0, Ctau)
    eigvals, eigvecs = np.linalg.eig(A)
    order = np.argsort(-np.abs(eigvals))
    W = np.real(eigvecs[:, order[: int(max(1, n_components))]])
    return cast(np.ndarray, Xc @ W)
