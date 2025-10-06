from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np

from .discretize import _coerce_array, _normalise_splits
from .project_cv import apply_whitening_from_metadata

logger = logging.getLogger(__name__)

DatasetLike = Mapping[str, Any] | MutableMapping[str, Any]
_TAU_SEQUENCE: tuple[int, ...] = (2, 5, 10, 20, 40)  # legacy base candidate lags (kept for backwards compatibility only)


class CanonicalCorrelationError(ValueError):
    """Base error raised when canonical correlation computation fails."""


class InsufficientSamplesError(CanonicalCorrelationError):
    """Raised when there are not enough paired samples (need at least 2)."""


# --- Canonical correlation helpers -------------------------------------------------

def _extract_optional_inputs(split: Mapping[str, Any]) -> np.ndarray | None:
    candidate_keys = (
        "inputs",
        "raw",
        "raw_inputs",
        "raw_features",
        "features",
        "input_features",
    )
    for key in candidate_keys:
        value = split.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            continue
        if arr.shape[0] == 0 or not np.isfinite(arr).all():
            continue
        return arr
    return None


def _validate_canonical_inputs(X: np.ndarray, Y: np.ndarray) -> int:
    """Validate inputs for canonical correlation returning usable sample size.

    Returns
    -------
    n : int
        Number of paired samples after truncation to the minimum length.
    Raises
    ------
    InsufficientSamplesError
        If fewer than 2 paired samples are available.
    CanonicalCorrelationError
        If shapes are incompatible (non-2D) or contain non-finite values.
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise CanonicalCorrelationError("X and Y must be 2D arrays")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise CanonicalCorrelationError("X and Y must contain only finite values")
    n = min(X.shape[0], Y.shape[0])
    if n < 2:
        logger.error("Canonical correlation: insufficient paired samples (n=%d < 2)", n)
        raise InsufficientSamplesError(f"Need at least 2 paired samples, got {n}")
    return n


def _center(X: np.ndarray) -> np.ndarray:
    return X - np.mean(X, axis=0, keepdims=True)


def _covariance(Xc: np.ndarray, n: int) -> np.ndarray:
    # Use unbiased denominator (n-1) with guard for numerical stability
    return (Xc.T @ Xc) / max(n - 1, 1)


def _inv_symmetric_sqrt(mat: np.ndarray, regularisation: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat + regularisation * np.eye(mat.shape[0]))
    eigvals = np.clip(eigvals, a_min=regularisation, a_max=None)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _canonical_correlations(
    X: np.ndarray, Y: np.ndarray, *, regularisation: float = 1e-8
) -> list[float]:
    """Compute canonical correlations between two 2D arrays.

    Implements a small, SOLID-style pipeline:
      * Validation (_validate_canonical_inputs)
      * Centering (_center)
      * Covariance computation (_covariance)
      * Whitening (_inv_symmetric_sqrt)
      * SVD decomposition (np.linalg.svd)

    Raises
    ------
    InsufficientSamplesError
        If fewer than two paired samples are available.
    CanonicalCorrelationError
        If numerical linear algebra fails (eigendecomposition / SVD) or inputs invalid.
    """
    n = _validate_canonical_inputs(X, Y)
    # Truncate to common length
    X = X[:n]
    Y = Y[:n]
    Xc = _center(X)
    Yc = _center(Y)
    Sxx = _covariance(Xc, n)
    Syy = _covariance(Yc, n)
    Sxy = (Xc.T @ Yc) / max(n - 1, 1)
    try:
        inv_sqrt_x = _inv_symmetric_sqrt(Sxx, regularisation)
        inv_sqrt_y = _inv_symmetric_sqrt(Syy, regularisation)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - rare numerical failure
        raise CanonicalCorrelationError(f"Eigen decomposition failed: {exc}") from exc
    M = inv_sqrt_x @ Sxy @ inv_sqrt_y
    try:
        _, s, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - rare numerical failure
        raise CanonicalCorrelationError(f"SVD failed: {exc}") from exc
    s = np.clip(s, 0.0, 1.0)
    return [float(val) for val in s]


# --- Autocorrelation helpers (refactored for SOLID) ------------------------------

def _validate_autocorr_input(X: np.ndarray) -> np.ndarray | None:
    """Validate and mean-center input for autocorrelation.

    Returns centered array or None if insufficient samples.
    Logs any issues instead of silent failure.
    """
    if X.ndim != 2:
        logger.warning("Autocorrelation: expected 2D array, got ndim=%d", X.ndim)
        return None
    if X.shape[0] < 2:
        logger.error("Autocorrelation: insufficient samples (n=%d < 2)", X.shape[0])
        return None
    if not np.isfinite(X).all():
        logger.warning("Autocorrelation: non-finite values detected; filtering may produce NaNs")
    return X - np.mean(X, axis=0, keepdims=True)


def _autocorr_at_lag(Xc: np.ndarray, tau: int) -> float:
    """Compute (mean across features) autocorrelation at a specific lag.

    Returns NaN if tau invalid or denominator zero.
    """
    n = Xc.shape[0]
    if tau <= 0 or tau >= n:
        logger.debug("Autocorrelation: invalid tau=%d for n=%d", tau, n)
        return float("nan")
    a = Xc[:-tau]
    b = Xc[tau:]
    numerator = np.sum(a * b, axis=0)
    denominator = np.sum(a * a, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
    if rho.size == 0:
        logger.debug("Autocorrelation: zero-size rho at tau=%d", tau)
        return float("nan")
    finite = rho[np.isfinite(rho)]
    if finite.size == 0:
        logger.debug("Autocorrelation: no finite rho values at tau=%d", tau)
        return float("nan")
    return float(np.mean(finite))


def _autocorrelation_curve(X: np.ndarray, taus: Sequence[int]) -> list[float]:
    """Compute autocorrelation curve across provided lags.

    Applies input validation, logs any anomalies, and delegates per-lag work to
    a focused helper for clearer Single Responsibility and easier testing.
    """
    Xc = _validate_autocorr_input(X)
    if Xc is None:
        logger.warning(
            "Autocorrelation: input invalid or insufficient; returning NaNs for all %d taus",
            len(taus),
        )
        return [float("nan") for _ in taus]
    curve: list[float] = []
    for tau in taus:
        curve.append(_autocorr_at_lag(Xc, tau))
    # Log if any NaNs present (surface potential silent degradation)
    nan_indices = [i for i, v in enumerate(curve) if not np.isfinite(v)]
    if nan_indices:
        logger.warning(
            "Autocorrelation: %d/%d NaN values in curve (first indices: %s)",
            len(nan_indices),
            len(curve),
            nan_indices[:10],
        )
    return curve


# --- Tau derivation / validation (core logic) -------------------------------------


def _validate_user_taus(user_taus: Sequence[int], min_length: int) -> list[int]:
    """Validate and sanitise user-provided tau list.

    Ensures:
      * All entries are ints >= 1.
      * Sorted strictly increasing (duplicates removed preserving order of first appearance).
      * At least one tau < min_length (otherwise we would produce only NaNs).

    Raises
    ------
    ValueError
        On any validation failure.
    """
    if not user_taus:
        raise ValueError("Provided taus sequence is empty")
    cleaned: list[int] = []
    seen: set[int] = set()
    last = 0
    for raw in user_taus:
        if not isinstance(raw, (int, np.integer)):
            raise ValueError(f"Tau '{raw}' is not an integer")
        t = int(raw)
        if t < 1:
            raise ValueError(f"Tau must be >=1, got {t}")
        if t in seen:
            continue  # deduplicate silently (explicit error would be noisy for common cases)
        if t <= last:
            raise ValueError("Taus must be strictly increasing")
        seen.add(t)
        cleaned.append(t)
        last = t
    if all(t >= min_length for t in cleaned):
        raise ValueError(
            f"All taus ({cleaned}) are >= minimum split length {min_length}; would yield all NaNs"
        )
    return cleaned


# --- Public tau API ----------------------------------------------------------------

def derive_taus(
    dataset: DatasetLike | Sequence[int],
    *,
    max_lags: int = 8,
    min_lag: int = 2,
    fraction_max: float = 0.5,
    geometric: bool = True,
    base: Sequence[int] | None = None,
) -> list[int]:
    """Derive a validated list of autocorrelation lag times (taus) dynamically.

    This replaces reliance on a hardcoded fixed set by computing a concise,
    information-spanning sequence of lag values tailored to the dataset.

    Strategy (default geometric=True):
      * Determine the minimum split length ``L`` across all splits.
      * Define an upper bound ``U = max(min_lag+1, floor(L * fraction_max))``.
      * Generate up to ``max_lags`` geometrically spaced integers between
        ``min_lag`` and ``U`` (exclusive of ``L``), ensuring strict increase
        and uniqueness.
      * Always enforce:  min_lag >= 1, max_lags >= 1, 0 < fraction_max <= 1.

    If ``geometric`` is False and ``base`` is provided, we validate the base
    list and filter to values < L (legacy compatibility path). If ``base`` is
    None and ``geometric`` is False, a ValueError is raised (explicit design –
    no silent fallbacks).

    Parameters
    ----------
    dataset : mapping-like OR sequence[int]
        Dataset with splits or a raw sequence of split lengths.
    max_lags : int, default 8
        Maximum number of lag values to attempt to generate.
    min_lag : int, default 2
        Smallest lag to consider (1 is allowed if explicitly set, but 0 never returned).
    fraction_max : float, default 0.5
        Fraction of the minimum split length used as an upper bound cap.
    geometric : bool, default True
        Whether to use geometric spacing. If False, requires a ``base`` sequence.
    base : sequence[int] | None, default None
        Optional legacy base list; if provided with geometric=True it will be
        ignored (a warning is logged). If provided with geometric=False it's
        validated & filtered below the minimum length.

    Returns
    -------
    list[int]
        Strictly increasing list of lags, each < minimum split length.

    Raises
    ------
    ValueError
        On invalid parameters, inability to determine lengths, or empty result.
    """
    # Validate parameter primitives early.
    if max_lags < 1:
        raise ValueError(f"max_lags must be >=1, got {max_lags}")
    if min_lag < 1:
        raise ValueError(f"min_lag must be >=1, got {min_lag}")
    if not (0 < fraction_max <= 1):
        raise ValueError(f"fraction_max must be in (0,1], got {fraction_max}")

    # Acquire split lengths.
    if isinstance(dataset, (Mapping, MutableMapping)) and not isinstance(dataset, (list, tuple)):
        splits = _normalise_splits(dataset)
        lengths: list[int] = []
        for value in splits.values():
            try:
                arr = _coerce_array(value)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Skipping split during tau derivation: %s", exc)
                continue
            lengths.append(int(arr.shape[0]))
    else:
        lengths = [int(l) for l in dataset]  # type: ignore[arg-type]

    if not lengths:
        raise ValueError("No split lengths available for tau derivation")
    if any(l <= 0 for l in lengths):
        raise ValueError(f"Non-positive split length encountered: {lengths}")

    min_len = min(lengths)
    if min_len <= min_lag:
        raise ValueError(
            f"Minimum split length {min_len} is not greater than min_lag {min_lag}; cannot derive taus."
        )

    if geometric:
        if base is not None:
            logger.warning("derive_taus: 'base' provided but ignored because geometric=True")
        upper_bound = int(max(min_lag + 1, np.floor(min_len * fraction_max)))
        # Ensure upper_bound < min_len.
        upper_bound = min(upper_bound, min_len - 1)
        if upper_bound <= min_lag:
            raise ValueError(
                f"Upper bound {upper_bound} not greater than min_lag {min_lag}; cannot derive taus."
            )
        # Geometric spacing: sample exponents uniformly in log-space.
        # Use np.linspace over log values inclusive of endpoints, then round & filter.
        start = np.log(min_lag)
        stop = np.log(upper_bound)
        raw = np.exp(np.linspace(start, stop, num=max_lags))
        # Convert to ints, enforce >= min_lag and < min_len.
        candidates = [int(round(v)) for v in raw]
        # Deduplicate while preserving order and enforce bounds & strict increase.
        taus: list[int] = []
        last = 0
        for t in candidates:
            if t < min_lag or t >= min_len:
                continue
            if t <= last:
                continue
            taus.append(t)
            last = t
        if not taus:
            raise ValueError(
                f"Geometric tau derivation yielded empty set (min_len={min_len}, min_lag={min_lag}, upper={upper_bound})."
            )
    else:
        if base is None:
            raise ValueError("Non-geometric tau derivation requires a 'base' sequence")
        if not base:
            raise ValueError("Base tau candidate sequence is empty")
        invalid = [b for b in base if (not isinstance(b, (int, np.integer))) or int(b) <= 0]
        if invalid:
            raise ValueError(f"Base tau sequence must contain only positive integers, got invalid entries {invalid}")
        taus = []
        seen: set[int] = set()
        for b in base:
            t = int(b)
            if t >= min_len or t in seen or t < min_lag:
                continue
            taus.append(t)
            seen.add(t)
        if not taus:
            raise ValueError(
                f"Base tau filtering produced empty set (base={list(base)}, min_len={min_len}, min_lag={min_lag})"
            )

    logger.info(
        "Derived taus %s (strategy=%s, min_length=%d, n_splits=%d)",
        taus,
        "geometric" if geometric else "base-filter",
        min_len,
        len(lengths),
    )
    return taus


# --- Public diagnostics API ------------------------------------------------------

def compute_diagnostics(
    dataset: DatasetLike,
    *,
    diag_mass: float | None = None,
    taus: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Compute triviality/stability diagnostics for downstream reporting.

    If ``taus`` is None, dynamically derive lag times using ``derive_taus``'s
    geometric heuristic. User-supplied ``taus`` are strictly validated.
    """
    splits = _normalise_splits(dataset)

    # Precompute split lengths for validation & tau logic.
    lengths: list[int] = []
    for value in splits.values():
        try:
            arr = _coerce_array(value)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skipping length collection for a split: %s", exc)
            continue
        lengths.append(int(arr.shape[0]))
    if not lengths:
        raise ValueError("Could not determine split lengths for tau derivation")
    min_length = min(lengths)

    if taus is None:
        taus_used = derive_taus(lengths)
    else:
        taus_used = _validate_user_taus(taus, min_length)
        logger.info("Validated user taus %s (min split length %d)", taus_used, min_length)

    canonical: Dict[str, list[float]] = {}
    autocorr: Dict[str, Dict[str, Any]] = {}
    warnings: list[str] = []

    for name, split in splits.items():
        processed = _compute_split_diagnostics(name, split, taus_used)
        if processed is None:
            continue
        split_canonical, split_autocorr, split_warnings = processed
        if split_canonical:
            canonical[name] = split_canonical
        autocorr[name] = split_autocorr
        warnings.extend(split_warnings)

    if diag_mass is not None and np.isfinite(diag_mass) and diag_mass > 0.95:
        msg = f"MSM diagonal mass high ({diag_mass:.3f})"
        warnings.append(msg)
        logger.warning(msg)

    return {
        "canonical_correlation": canonical,
        "autocorrelation": autocorr,
        "diag_mass": float(diag_mass) if diag_mass is not None else None,
        "taus": list(taus_used),
        "warnings": warnings,
    }


def _compute_split_diagnostics(
    name: str,
    split: Any,
    taus: Sequence[int],
) -> tuple[list[float] | None, Dict[str, Any], list[str]] | None:
    """Gather canonical correlations and autocorrelation curve for one split."""

    try:
        X = _coerce_array(split)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Skipping diagnostic split %s: %s", name, exc)
        return None

    metadata = split.get("meta") if isinstance(split, Mapping) else None
    whitened, _ = apply_whitening_from_metadata(X, metadata)

    canonical: list[float] | None = None
    warnings: list[str] = []
    inputs = _extract_optional_inputs(split) if isinstance(split, Mapping) else None
    if inputs is not None:
        if inputs.shape[0] != whitened.shape[0]:
            length = min(inputs.shape[0], whitened.shape[0])
            logger.debug(
                "Truncating inputs/whitened for canonical correlation: %s length -> %d",
                name,
                length,
            )
            inputs = inputs[:length]
            whitened = whitened[:length]
        try:
            correlations = _canonical_correlations(inputs, whitened)
        except InsufficientSamplesError:
            logger.error(
                "%s: insufficient samples for canonical correlation (need >=2)", name
            )
            # Propagate as per requirement to raise error on insufficient samples
            raise
        except CanonicalCorrelationError as exc:
            msg = f"{name}: canonical correlation failed ({exc})"
            warnings.append(msg)
            logger.warning(msg)
            correlations = []
        else:
            if correlations:
                canonical = correlations
                if min(correlations) > 0.95:
                    msg = f"{name}: CVs reparametrize inputs"
                    warnings.append(msg)
                    logger.warning(msg)

    curve = _autocorrelation_curve(whitened, taus)
    autocorr = {"taus": list(taus), "values": curve}
    if len(curve) >= 4 and np.isfinite(curve[0]) and np.isfinite(curve[3]):
        if abs(curve[0] - curve[3]) < 0.05:
            msg = f"{name}: CV autocorrelation flat across lags"
            warnings.append(msg)
            logger.warning(msg)

    return canonical, autocorr, warnings
