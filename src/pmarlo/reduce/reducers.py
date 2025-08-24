from __future__ import annotations

from typing import List, Optional, cast

import numpy as np


def _preprocess(X: np.ndarray, scale: bool = True) -> np.ndarray:
    """Center and optionally scale features in a NaN-safe manner."""
    Xp = np.asarray(X, dtype=float)
    mean = np.nanmean(Xp, axis=0, keepdims=True)
    mean = cast(np.ndarray, np.nan_to_num(mean, nan=0.0))
    Xp = cast(np.ndarray, np.nan_to_num(Xp - mean, nan=0.0))
    if scale:
        std = np.nanstd(Xp, axis=0, keepdims=True)
        std = np.nan_to_num(std, nan=1.0)
        std[std == 0] = 1.0
        Xp = Xp / std
    return np.nan_to_num(Xp, nan=0.0)


def pca_reduce(
    X: np.ndarray,
    n_components: int = 2,
    batch_size: Optional[int] = None,
    scale: bool = True,
) -> np.ndarray:
    """PCA reduction with optional batching and feature scaling.

    Parameters
    ----------
    X: array-like, shape (n_frames, n_features)
        Input data.
    n_components: int, default=2
        Number of principal components to keep.
    batch_size: Optional[int]
        If provided, use incremental PCA processing in chunks of this size.
    scale: bool, default=True
        When True, scale features to unit variance in addition to centering.

    Returns
    -------
    np.ndarray, shape (n_frames, n_components)
        Projected data.
    """
    if X.size == 0:
        return np.zeros((X.shape[0], n_components), dtype=float)
    Xp = _preprocess(X, scale=scale)
    n_comp = int(max(1, n_components))
    if batch_size is None or Xp.shape[0] <= batch_size:
        # Economy SVD
        U, S, Vt = np.linalg.svd(Xp, full_matrices=False)
        k = int(min(n_comp, Vt.shape[0]))
        return cast(np.ndarray, U[:, :k] @ np.diag(S[:k]))
    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=n_comp)
    for start in range(0, Xp.shape[0], batch_size):
        ipca.partial_fit(Xp[start : start + batch_size])
    return cast(np.ndarray, ipca.transform(Xp))


def tica_reduce(
    X: np.ndarray,
    lag: int = 10,
    n_components: int = 2,
    batch_size: Optional[int] = None,
    scale: bool = True,
) -> np.ndarray:
    """TICA reduction with optional batching and feature scaling.

    Falls back to an internal generalized eigenvalue solver when deeptime is
    unavailable. The input is always centered and optionally scaled in a
    NaN-safe manner.
    """
    if X.size == 0:
        return np.zeros((X.shape[0], n_components), dtype=float)
    Xp = _preprocess(X, scale=scale)
    # Prefer deeptime implementation for numerical stability and lag-aware behavior
    try:
        from deeptime.decomposition import TICA as _DT_TICA  # type: ignore

        series: List[np.ndarray] = [Xp]
        dim = int(max(1, n_components))
        model = _DT_TICA(lagtime=int(max(1, lag)), dim=dim).fit(series).fetch_model()
        Y_list = model.transform(series)
        Y = np.asarray(Y_list[0], dtype=float)
        return _ensure_component_count(Y, dim)
    except Exception:
        return _tica_fallback(Xp, lag, n_components, batch_size)


def _tica_fallback(
    X: np.ndarray, lag: int, n_components: int, batch_size: Optional[int]
) -> np.ndarray:
    """Internal TICA solver with batching support."""
    if X.shape[0] <= lag + 1:
        return pca_reduce(X, n_components=n_components)
    C0 = np.zeros((X.shape[1], X.shape[1]))
    Ctau = np.zeros_like(C0)
    bs = batch_size or (X.shape[0] - lag)
    for start in range(0, X.shape[0] - lag, bs):
        end = min(X.shape[0] - lag, start + bs)
        X0 = X[start:end]
        X1 = X[start + lag : end + lag]
        C0 += X0.T @ X0
        Ctau += X0.T @ X1
    eps = 1e-6
    C0 += eps * np.eye(C0.shape[0])
    A = np.linalg.solve(C0, Ctau)
    eigvals, eigvecs = np.linalg.eig(A)
    order = np.argsort(-np.abs(eigvals))
    W = np.real(eigvecs[:, order[: int(max(1, n_components))]])
    return cast(np.ndarray, X @ W)


def vamp_reduce(
    X: np.ndarray,
    lag: int = 10,
    n_components: int = 2,
    score_dims: Optional[List[int]] = None,
    scale: bool = True,
) -> np.ndarray:
    """VAMP reduction using deeptime with optional dimension selection.

    Input data are centered and optionally scaled in a NaN-safe fashion. Falls
    back to PCA if deeptime is unavailable or errors.
    """
    if X.size == 0:
        return np.zeros((X.shape[0], n_components), dtype=float)
    Xp = _preprocess(X, scale=scale)
    series: List[np.ndarray] = [Xp]
    dim = int(max(1, n_components))

    # Try deeptime VAMP transform path
    try:
        dim = _vamp_select_dimension(series, lag, dim, score_dims)
        Y = _vamp_transform(series, lag, dim)
        return _ensure_component_count(Y, dim)
    except Exception:
        return pca_reduce(Xp, n_components=n_components)


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
