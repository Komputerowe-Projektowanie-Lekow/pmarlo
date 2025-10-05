from __future__ import annotations

import importlib.util
from typing import List, Optional, cast

import numpy as np

try:  # pragma: no cover - optional SciPy dependency
    from scipy.linalg import eigh as scipy_eigh
except Exception:  # pragma: no cover - SciPy optional
    scipy_eigh = None


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
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    n_components : int
        Number of principal components to retain.
    batch_size : Optional[int]
        If provided, uses mini-batch PCA. Otherwise uses standard PCA.
    scale : bool
        Whether to standardize features before PCA.

    Returns
    -------
    np.ndarray
        Transformed data (n_samples, n_components).
    """
    X_prep = _preprocess(X, scale=scale)

    # Try sklearn first
    try:
        from sklearn.decomposition import PCA, IncrementalPCA

        if batch_size is not None:
            pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        else:
            pca = PCA(n_components=n_components)
        return pca.fit_transform(X_prep)
    except ImportError:
        pass

    # Fallback to numpy SVD
    U, S, Vt = np.linalg.svd(X_prep, full_matrices=False)
    return U[:, :n_components] * S[:n_components]


def tica_reduce(
    X: np.ndarray,
    lag: int = 1,
    n_components: int = 2,
    scale: bool = True,
) -> np.ndarray:
    """TICA (Time-lagged Independent Component Analysis) reduction.

    Parameters
    ----------
    X : np.ndarray
        Time series feature matrix (n_frames, n_features).
    lag : int
        Lag time for time-lagged correlations.
    n_components : int
        Number of independent components to retain.
    scale : bool
        Whether to standardize features before TICA.

    Returns
    -------
    np.ndarray
        TICA-transformed data (n_frames, n_components).
    """
    X_prep = _preprocess(X, scale=scale)

    # Try deeptime first (preferred for TICA)
    try:
        from deeptime.decomposition import TICA

        tica = TICA(lagtime=lag, dim=n_components)
        model = tica.fit(X_prep)
        return model.transform(X_prep)
    except ImportError:
        pass

    # Try pyemma as fallback
    try:
        import pyemma

        tica = pyemma.coordinates.tica([X_prep], lag=lag, dim=n_components)
        return tica.get_output()[0]
    except ImportError:
        pass

    # Simple manual TICA implementation as last resort
    return _manual_tica(X_prep, lag=lag, n_components=n_components)


def _manual_tica(X: np.ndarray, lag: int = 1, n_components: int = 2) -> np.ndarray:
    """Manual TICA implementation using generalized eigenvalue problem."""
    n_frames = X.shape[0]
    if lag >= n_frames:
        raise ValueError(
            f"Lag time {lag} must be less than number of frames {n_frames}"
        )

    # Compute covariance matrices
    X_t = X[:-lag]  # X(t)
    X_t_lag = X[lag:]  # X(t+lag)

    # Instantaneous covariance C_00
    C_00 = np.cov(X_t.T)

    # Time-lagged covariance C_0t
    C_0t = np.cov(X_t.T, X_t_lag.T)[: X.shape[1], X.shape[1] :]

    # Solve generalized eigenvalue problem
    try:
        if scipy_eigh is not None:
            eigenvals, eigenvecs = scipy_eigh(C_0t @ C_0t.T, C_00)
        else:
            inv_c00 = np.linalg.pinv(C_00)
            mat = inv_c00 @ (C_0t @ C_0t.T)
            eigenvals, eigenvecs = np.linalg.eigh(mat)
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx]

        # Transform data
        return X @ eigenvecs[:, :n_components]
    except np.linalg.LinAlgError:
        # Fallback to PCA if TICA fails
        return pca_reduce(X, n_components=n_components, scale=False)


def vamp_reduce(
    X: np.ndarray,
    lag: int = 1,
    n_components: int = 2,
    scale: bool = True,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """VAMP (Variational Approach for Markov Processes) reduction.

    Parameters
    ----------
    X : np.ndarray
        Time series feature matrix (n_frames, n_features).
    lag : int
        Lag time for transition analysis.
    n_components : int
        Number of VAMP components to retain.
    scale : bool
        Whether to standardize features before VAMP.
    epsilon : float
        Regularization parameter for numerical stability.

    Returns
    -------
    np.ndarray
        VAMP-transformed data (n_frames, n_components).
    """
    X_prep = _preprocess(X, scale=scale)

    # Try deeptime first (preferred for VAMP)
    try:
        from deeptime.decomposition import VAMP

        vamp = VAMP(lagtime=lag, dim=n_components, epsilon=epsilon)
        model = vamp.fit([X_prep])
        return model.transform(X_prep)
    except ImportError:
        pass

    # Try pyemma as fallback
    try:
        import pyemma

        vamp = pyemma.coordinates.vamp([X_prep], lag=lag, dim=n_components)
        return vamp.get_output()[0]
    except ImportError:
        pass

    # Manual VAMP implementation as last resort
    return _manual_vamp(X_prep, lag=lag, n_components=n_components, epsilon=epsilon)


def _manual_vamp(
    X: np.ndarray, lag: int = 1, n_components: int = 2, epsilon: float = 1e-6
) -> np.ndarray:
    """Manual VAMP implementation using SVD-based approach."""
    n_frames = X.shape[0]
    if lag >= n_frames:
        raise ValueError(
            f"Lag time {lag} must be less than number of frames {n_frames}"
        )

    # Split data
    X_t = X[:-lag]  # X(t)
    X_t_lag = X[lag:]  # X(t+lag)

    # Compute covariance matrices with regularization
    C_00 = np.cov(X_t.T) + epsilon * np.eye(X.shape[1])
    C_11 = np.cov(X_t_lag.T) + epsilon * np.eye(X.shape[1])
    C_01 = np.cov(X_t.T, X_t_lag.T)[: X.shape[1], X.shape[1] :]

    try:
        # VAMP-2 score optimization via SVD
        # K = C_00^{-1/2} C_01 C_11^{-1/2}
        L_00 = np.linalg.cholesky(C_00)
        L_11 = np.linalg.cholesky(C_11)

        K = np.linalg.solve(L_00, C_01)
        K = np.linalg.solve(L_11.T, K.T).T

        U, s, Vt = np.linalg.svd(K, full_matrices=False)

        # Transform functions
        psi = np.linalg.solve(L_00, U[:, :n_components].T).T

        # Apply transformation
        return X @ psi
    except np.linalg.LinAlgError:
        # Fallback to TICA if VAMP fails
        return tica_reduce(X, lag=lag, n_components=n_components, scale=False)


# Convenience functions for common use cases
def reduce_features(
    X: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    lag: int = 1,
    scale: bool = True,
    **kwargs,
) -> np.ndarray:
    """Unified interface for dimensionality reduction.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    method : str
        Reduction method: "pca", "tica", or "vamp".
    n_components : int
        Number of components to retain.
    lag : int
        Lag time (for TICA/VAMP only).
    scale : bool
        Whether to standardize features.
    **kwargs
        Additional method-specific parameters.

    Returns
    -------
    np.ndarray
        Reduced feature matrix.
    """
    method = method.lower()

    if method == "pca":
        return pca_reduce(X, n_components=n_components, scale=scale, **kwargs)
    elif method == "tica":
        return tica_reduce(X, lag=lag, n_components=n_components, scale=scale, **kwargs)
    elif method == "vamp":
        return vamp_reduce(X, lag=lag, n_components=n_components, scale=scale, **kwargs)
    else:
        raise ValueError(f"Unknown reduction method: {method}")


def get_available_methods() -> List[str]:
    """Get list of available reduction methods based on installed packages.

    Returns
    -------
    List[str]
        List of available methods.
    """
    methods = ["pca"]  # Always available (numpy fallback)

    # Check for deeptime
    if importlib.util.find_spec("deeptime") is not None:
        methods.extend(["tica", "vamp"])

    # Check for pyemma (fallback)
    if importlib.util.find_spec("pyemma") is not None:
        if "tica" not in methods:
            methods.append("tica")
        if "vamp" not in methods:
            methods.append("vamp")

    # Manual implementations always available as fallback
    if "tica" not in methods:
        methods.append("tica")
    if "vamp" not in methods:
        methods.append("vamp")

    return methods
