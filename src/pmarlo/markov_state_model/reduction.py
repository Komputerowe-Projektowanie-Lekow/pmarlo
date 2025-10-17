from __future__ import annotations

import importlib.util
from typing import List, Optional, cast

import numpy as np

from pmarlo import constants as const

try:  # pragma: no cover - optional SciPy dependency
    from scipy.linalg import eigh as scipy_eigh, solve_triangular
except Exception:  # pragma: no cover - SciPy optional
    scipy_eigh = None
    solve_triangular = None

try:  # pragma: no cover - scikit-learn is an optional heavy dependency at runtime
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - keep working if scikit-learn missing
    SimpleImputer = None
    StandardScaler = None


def _nan_safe_manual_preprocess(X: np.ndarray, scale: bool) -> np.ndarray:
    """Fallback preprocessing using the previous manual implementation."""

    mean = np.nanmean(X, axis=0, keepdims=True)
    mean = cast(np.ndarray, np.nan_to_num(mean, nan=0.0))
    centered = cast(np.ndarray, np.nan_to_num(X - mean, nan=0.0))
    if scale:
        std = np.nanstd(centered, axis=0, keepdims=True)
        std = np.nan_to_num(std, nan=1.0)
        std[std == 0] = 1.0
        centered = centered / std
    return np.nan_to_num(centered, nan=0.0)


def _preprocess(X: np.ndarray, scale: bool = True) -> np.ndarray:
    """Center and optionally scale features using scikit-learn with NaN handling."""

    Xp = np.asarray(X, dtype=float)

    if Xp.size == 0:
        return cast(np.ndarray, np.zeros_like(Xp, dtype=float))

    # ``StandardScaler`` requires 2D input. Preserve the original dimensionality
    # so callers receive the same shape that they passed in.
    squeeze_1d = False
    if Xp.ndim == 1:
        Xp = Xp.reshape(-1, 1)
        squeeze_1d = True

    if SimpleImputer is None or StandardScaler is None:
        result = _nan_safe_manual_preprocess(Xp, scale=scale)
    else:
        imputer = SimpleImputer(strategy="mean")
        try:
            X_imputed = imputer.fit_transform(Xp)
        except ValueError:
            # Columns that are entirely NaN raise ``ValueError`` for ``strategy='mean'``.
            # Fall back to a constant fill that mirrors the legacy behaviour of
            # treating missing values as zeros.
            imputer = SimpleImputer(strategy="constant", fill_value=0.0)
            X_imputed = imputer.fit_transform(Xp)

        scaler = StandardScaler(with_mean=True, with_std=scale)
        try:
            result = scaler.fit_transform(X_imputed)
        except ValueError:
            # If scikit-learn encounters an edge case (e.g. a single sample), use the
            # stable manual routine as a safety net.
            result = _nan_safe_manual_preprocess(X_imputed, scale=scale)
        else:
            result = np.nan_to_num(result, nan=0.0)

    if squeeze_1d:
        result = result.reshape(-1)
    return cast(np.ndarray, result)


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

def _covariance_block(
    X_t: np.ndarray,
    X_t_lag: np.ndarray,
    *,
    regularization: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute covariance and cross-covariance matrices via :func:`numpy.cov`.

    Parameters
    ----------
    X_t, X_t_lag:
        Arrays containing aligned samples ``(n_frames, n_features)`` representing
        ``X(t)`` and ``X(t + lag)`` respectively.
    regularization:
        Optional diagonal term added to the auto-covariance blocks.  This mirrors
        the behaviour of the manual fallbacks which previously sprinkled
        ``epsilon`` to stabilise inverses.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(C_00, C_0t, C_tt)`` where ``C_00`` and ``C_tt`` are the auto-covariance
        matrices for ``X_t`` and ``X_t_lag`` and ``C_0t`` is the cross-covariance
        between them.
    """

    X_t = np.asarray(X_t, dtype=float)
    X_t_lag = np.asarray(X_t_lag, dtype=float)
    n_features = X_t.shape[1]

    # ``np.cov`` expects observations along rows when ``rowvar=False``.
    stacked = np.hstack((X_t, X_t_lag))
    cov_full = np.cov(stacked, rowvar=False)

    C_00 = np.array(cov_full[:n_features, :n_features], copy=False)
    C_0t = np.array(cov_full[:n_features, n_features:], copy=False)
    C_tt = np.array(cov_full[n_features:, n_features:], copy=False)

    if regularization:
        reg = regularization * np.eye(n_features)
        C_00 = C_00 + reg
        C_tt = C_tt + reg

    return C_00, C_0t, C_tt


def _lagged_covariances(
    X: np.ndarray, lag: int, *, epsilon: float = 0.0, ddof: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lagged covariance blocks computed via :func:`numpy.cov`."""

    X_t = X[:-lag]
    X_t_lag = X[lag:]
    n_features = X.shape[1]
    if X_t.shape[0] <= 1:
        zeros = np.zeros((n_features, n_features), dtype=X.dtype)
        if epsilon:
            eye = np.eye(n_features, dtype=X.dtype)
            return zeros + epsilon * eye, zeros + epsilon * eye, zeros
        return zeros, zeros, zeros

    combined = np.concatenate((X_t, X_t_lag), axis=1)
    cov = np.cov(combined, rowvar=False, ddof=ddof)
    C_00 = cov[:n_features, :n_features]
    C_11 = cov[n_features:, n_features:]
    C_01 = cov[:n_features, n_features:]
    if epsilon:
        eye = np.eye(n_features, dtype=C_00.dtype)
        C_00 = C_00 + epsilon * eye
        C_11 = C_11 + epsilon * eye
    return C_00, C_11, C_01


def _generalized_eigh(
    A: np.ndarray,
    B: np.ndarray,
    *,
    epsilon: float = const.NUMERIC_ABSOLUTE_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the symmetric generalised eigenproblem ``A v = \lambda B v``.

    ``B`` is regularised to remain positive definite which improves numerical
    stability for the manual TICA fallback when covariance blocks are close to
    singular.
    """

    # Ensure symmetry because numerical noise from ``np.cov`` may introduce
    # slight asymmetries that trip the Hermitian solvers used below.
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)

    if epsilon:
        B = B + epsilon * np.eye(B.shape[0], dtype=B.dtype)

    if scipy_eigh is not None:
        return scipy_eigh(A, B, check_finite=False)

    # ``B`` is symmetric positive-definite after the regularisation above, so we
    # can whiten the problem and fall back to :func:`numpy.linalg.eigh`.
    evals_B, evecs_B = np.linalg.eigh(B)
    positive = evals_B > epsilon
    if not np.any(positive):
        raise np.linalg.LinAlgError("Covariance matrix is singular")

    whitening = evecs_B[:, positive] * (evals_B[positive] ** -0.5)
    A_tilde = whitening.T @ (A @ whitening)
    A_tilde = 0.5 * (A_tilde + A_tilde.T)

    evals, evecs = np.linalg.eigh(A_tilde)
    return evals, whitening @ evecs


def _manual_tica(X: np.ndarray, lag: int = 1, n_components: int = 2) -> np.ndarray:
    """Manual TICA implementation using generalized eigenvalue problem."""

    n_frames = X.shape[0]
    if lag >= n_frames:
        raise ValueError(
            f"Lag time {lag} must be less than number of frames {n_frames}"
        )

    C_00, _, C_0t = _lagged_covariances(X, lag, ddof=None)

    try:
        eigenvals, eigenvecs = _generalized_eigh(C_0t, C_00)
    except np.linalg.LinAlgError:
        # Fallback to PCA if TICA fails
        return pca_reduce(X, n_components=n_components, scale=False)

    # Sort components by the absolute eigenvalue magnitude (closer to 1 is better)
    idx = np.argsort(np.abs(eigenvals))[::-1]
    eigenvecs = np.asarray(np.real(eigenvecs[:, idx]), dtype=float)

    # ``n_components`` may be larger than the available eigenvectors when the
    # covariance matrix is rank-deficient.
    n_available = eigenvecs.shape[1]
    n_keep = min(n_components, n_available)
    if n_keep == 0:
        return pca_reduce(X, n_components=n_components, scale=False)

    return X @ eigenvecs[:, :n_keep]


def vamp_reduce(
    X: np.ndarray,
    lag: int = 1,
    n_components: int = 2,
    scale: bool = True,
    epsilon: float = const.NUMERIC_ABSOLUTE_TOLERANCE,
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
    X: np.ndarray,
    lag: int = 1,
    n_components: int = 2,
    epsilon: float = const.NUMERIC_ABSOLUTE_TOLERANCE,
) -> np.ndarray:
    """Manual VAMP implementation using SVD-based approach."""
    n_frames = X.shape[0]
    if lag >= n_frames:
        raise ValueError(
            f"Lag time {lag} must be less than number of frames {n_frames}"
        )

    C_00, C_11, C_01 = _lagged_covariances(X, lag, epsilon=epsilon, ddof=None)

    try:
        # VAMP-2 score optimization via SVD
        # K = C_00^{-1/2} C_01 C_11^{-1/2}
        L_00 = np.linalg.cholesky(C_00)
        L_11 = np.linalg.cholesky(C_11)

        if solve_triangular is not None:
            K = solve_triangular(L_00, C_01, lower=True, check_finite=False)
            K = solve_triangular(
                L_11,
                K.T,
                lower=True,
                trans="T",
                check_finite=False,
            ).T
        else:
            K = np.linalg.solve(L_00, C_01)
            K = np.linalg.solve(L_11.T, K.T).T

        U, s, Vt = np.linalg.svd(K, full_matrices=False)

        # Transform functions
        if solve_triangular is not None:
            psi = solve_triangular(
                L_00,
                U[:, :n_components],
                lower=True,
                check_finite=False,
            )
        else:
            psi = np.linalg.solve(L_00, U[:, :n_components])

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
