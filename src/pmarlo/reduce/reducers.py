from __future__ import annotations

from typing import List, Optional, cast

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
    """TICA using deeptime when available; falls back to internal solver.

    - Returns projected data of shape (n_frames, n_components)
    - Accepts a single time series X (n_frames, n_features)
    """
    if X.size == 0:
        return np.zeros((X.shape[0], n_components), dtype=float)
    # Prefer deeptime implementation for numerical stability and lag-aware behavior
    try:
        from deeptime.decomposition import TICA as _DT_TICA  # type: ignore

        series: List[np.ndarray] = [np.asarray(X, dtype=float)]
        dim = int(max(1, n_components))
        model = _DT_TICA(lagtime=int(max(1, lag)), dim=dim).fit(series).fetch_model()
        Y_list = model.transform(series)
        Y = np.asarray(Y_list[0], dtype=float)
        # Ensure exact component count
        if Y.shape[1] != dim:
            Y = (
                Y[:, :dim]
                if Y.shape[1] > dim
                else np.pad(Y, ((0, 0), (0, dim - Y.shape[1])), mode="constant")
            )
        return Y
    except Exception:
        # Fallback: lightweight generalized eigenvalue approach
        Xc = X - np.mean(X, axis=0, keepdims=True)
        if Xc.shape[0] <= lag + 1:
            return pca_reduce(Xc, n_components=n_components)
        X0 = Xc[:-lag]
        X1 = Xc[lag:]
        C0 = X0.T @ X0
        Ctau = X0.T @ X1
        eps = 1e-6
        C0 += eps * np.eye(C0.shape[0])
        A = np.linalg.solve(C0, Ctau)
        eigvals, eigvecs = np.linalg.eig(A)
        order = np.argsort(-np.abs(eigvals))
        W = np.real(eigvecs[:, order[: int(max(1, n_components))]])
        return cast(np.ndarray, Xc @ W)


def vamp_reduce(
    X: np.ndarray,
    lag: int = 10,
    n_components: int = 2,
    score_dims: Optional[List[int]] = None,
) -> np.ndarray:
    """VAMP reduction using deeptime with optional dimension selection.

    Falls back to PCA if deeptime is unavailable or errors.
    """
    if X.size == 0:
        return np.zeros((X.shape[0], n_components), dtype=float)
    series: List[np.ndarray] = [np.asarray(X, dtype=float)]
    dim = int(max(1, n_components))

    # Try deeptime VAMP transform path
    try:
        dim = _vamp_select_dimension(series, lag, dim, score_dims)
        Y = _vamp_transform(series, lag, dim)
        return _ensure_component_count(Y, dim)
    except Exception:
        return pca_reduce(X, n_components=n_components)


def _vamp_select_dimension(
    series: List[np.ndarray],
    lag: int,
    default_dim: int,
    score_dims: Optional[List[int]],
) -> int:
    if not score_dims:
        return default_dim
    try:
        from deeptime.decomposition import VAMP as _DT_VAMP  # type: ignore

        best_dim = None
        best_score = -np.inf
        for d in sorted({int(max(1, v)) for v in score_dims}):
            model = _DT_VAMP(lagtime=int(max(1, lag)), dim=d).fit(series).fetch_model()
            try:
                score = float(model.score(series))  # type: ignore[attr-defined]
            except Exception:
                score = -np.inf
            if score > best_score:
                best_score = score
                best_dim = d
        return int(best_dim if best_dim is not None else default_dim)
    except Exception:
        return default_dim


def _vamp_transform(series: List[np.ndarray], lag: int, dim: int) -> np.ndarray:
    from deeptime.decomposition import VAMP as _DT_VAMP  # type: ignore

    model = _DT_VAMP(lagtime=int(max(1, lag)), dim=int(max(1, dim))).fit(series)
    fetched = model.fetch_model()
    Y_list = fetched.transform(series)
    return np.asarray(Y_list[0], dtype=float)


def _ensure_component_count(Y: np.ndarray, dim: int) -> np.ndarray:
    if Y.shape[1] == dim:
        return Y
    if Y.shape[1] > dim:
        return Y[:, :dim]
    pad = dim - Y.shape[1]
    return np.pad(Y, ((0, 0), (0, pad)), mode="constant")
