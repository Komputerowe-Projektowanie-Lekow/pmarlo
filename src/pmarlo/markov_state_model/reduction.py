from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from pmarlo import constants as const


def _preprocess(X: np.ndarray, scale: bool = True) -> np.ndarray:
    """Center and optionally scale features using scikit-learn with NaN handling."""

    Xp = np.asarray(X, dtype=float)

    if Xp.size == 0:
        return np.zeros_like(Xp, dtype=float)

    squeeze_1d = False
    if Xp.ndim == 1:
        Xp = Xp.reshape(-1, 1)
        squeeze_1d = True

    imputer = SimpleImputer(strategy="mean")
    try:
        X_imputed = imputer.fit_transform(Xp)
    except ValueError:
        # ``SimpleImputer`` cannot compute the mean for all-NaN columns; mirror the
        # historical behaviour by replacing those columns with zeros instead.
        X_imputed = SimpleImputer(strategy="constant", fill_value=0.0).fit_transform(Xp)

    scaler = StandardScaler(with_mean=True, with_std=scale)
    result = scaler.fit_transform(X_imputed)
    result = np.nan_to_num(result, nan=0.0)

    if squeeze_1d:
        result = result.reshape(-1)
    return result


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

    if batch_size is not None:
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    else:
        pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(X_prep)
    return np.asarray(transformed, dtype=float)


def tica_reduce(
    X: np.ndarray,
    lag: int = 1,
    n_components: int = 2,
    scale: bool = True,
) -> np.ndarray:
    """TICA (Time-lagged Independent Component Analysis) reduction.

    Requires deeptime library to be installed.

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
    from deeptime.decomposition import TICA  # type: ignore

    X_prep = _preprocess(X, scale=scale)
    tica = TICA(lagtime=lag, dim=n_components)
    # deeptime's TICA.fit() expects a list of trajectories
    model = tica.fit([X_prep]).fetch_model()
    transformed = model.transform(X_prep)
    return np.asarray(transformed, dtype=float)


def vamp_reduce(
    X: np.ndarray,
    lag: int = 1,
    n_components: int = 2,
    scale: bool = True,
    epsilon: float = const.NUMERIC_ABSOLUTE_TOLERANCE,
) -> np.ndarray:
    """VAMP (Variational Approach for Markov Processes) reduction.

    Requires deeptime library to be installed.

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
    from deeptime.decomposition import VAMP  # type: ignore

    X_prep = _preprocess(X, scale=scale)
    vamp = VAMP(lagtime=lag, dim=n_components, epsilon=epsilon)
    model = vamp.fit([X_prep])
    transformed = model.transform(X_prep)
    return np.asarray(transformed, dtype=float)


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
    """Get list of available reduction methods.

    Returns
    -------
    List[str]
        List of available methods: ['pca', 'tica', 'vamp'].
        Note: TICA and VAMP require deeptime library.
    """
    return ["pca", "tica", "vamp"]
