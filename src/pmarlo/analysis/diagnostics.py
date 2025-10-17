from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np
from sklearn.cross_decomposition import CCA

from pmarlo import constants as const

from .discretize import _coerce_array, _normalise_splits
from .project_cv import apply_whitening_from_metadata

logger = logging.getLogger(__name__)

DatasetLike = Mapping[str, Any] | MutableMapping[str, Any]
_TAU_SEQUENCE: tuple[int, ...] = (
    2,
    5,
    10,
    20,
    40,
)  # legacy base candidate lags (kept for backwards compatibility only)


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
    n = min(int(X.shape[0]), int(Y.shape[0]))
    if n < 2:
        logger.error("Canonical correlation: insufficient paired samples (n=%d < 2)", n)
        raise InsufficientSamplesError(f"Need at least 2 paired samples, got {n}")
    return n


def _center(X: np.ndarray) -> np.ndarray:
    return X - np.mean(X, axis=0, keepdims=True)


def _covariance(Xc: np.ndarray, n: int) -> np.ndarray:
    """Delegate covariance estimation to :func:`numpy.cov` for robustness."""

    if Xc.size == 0:
        return np.zeros((Xc.shape[1], Xc.shape[1]), dtype=Xc.dtype)
    if n <= 1:
        return np.zeros((Xc.shape[1], Xc.shape[1]), dtype=Xc.dtype)
    return np.cov(Xc, rowvar=False, ddof=1)


def _inv_symmetric_sqrt(mat: np.ndarray, regularisation: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat + regularisation * np.eye(mat.shape[0]))
    eigvals = np.clip(eigvals, a_min=regularisation, a_max=None)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

def _canonical_correlations(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    regularisation: float = const.NUMERIC_RELATIVE_TOLERANCE,
) -> list[float]:
    """Compute canonical correlations between two 2D arrays.

    Raises
    ------
    InsufficientSamplesError
        If fewer than two paired samples are available.
    CanonicalCorrelationError
        If numerical linear algebra fails (sklearn CCA) or inputs invalid.
    """
    n = _validate_canonical_inputs(X, Y)
    # Truncate to common length
    X = X[:n]
    Y = Y[:n]
    n_components = min(X.shape[1], Y.shape[1], n)
    if n_components <= 0:
        return []
    try:
        cca = CCA(n_components=n_components, scale=False, max_iter=5000)
        X_c, Y_c = cca.fit_transform(X, Y)
    except ValueError as exc:  # pragma: no cover - validation/fit failures are rare
        raise CanonicalCorrelationError(f"CCA fitting failed: {exc}") from exc
    correlations: list[float] = []
    for idx in range(n_components):
        x_comp = X_c[:, idx]
        y_comp = Y_c[:, idx]
        numerator = float(np.dot(x_comp, y_comp))
        denominator = float(np.linalg.norm(x_comp) * np.linalg.norm(y_comp))
        if denominator <= regularisation:
            corr = 0.0
        else:
            corr = numerator / denominator
        correlations.append(float(np.clip(abs(corr), 0.0, 1.0)))
    return correlations


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
        logger.warning(
            "Autocorrelation: non-finite values detected; filtering may produce NaNs"
        )
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
    """Return a validated list of autocorrelation lag times for the dataset."""

    _validate_tau_parameters(
        max_lags=max_lags,
        min_lag=min_lag,
        fraction_max=fraction_max,
    )

    lengths = _collect_tau_lengths(dataset)
    min_length = _ensure_minimum_length(lengths, min_lag=min_lag)

    if geometric:
        taus = _derive_geometric_taus(
            min_length=min_length,
            min_lag=min_lag,
            fraction_max=fraction_max,
            max_lags=max_lags,
            base=base,
        )
        strategy = "geometric"
    else:
        taus = _derive_base_taus(
            base=base,
            min_length=min_length,
            min_lag=min_lag,
        )
        strategy = "base-filter"

    logger.info(
        "Derived taus %s (strategy=%s, min_length=%d, n_splits=%d)",
        taus,
        strategy,
        min_length,
        len(lengths),
    )
    return taus


def _validate_tau_parameters(
    *, max_lags: int, min_lag: int, fraction_max: float
) -> None:
    if max_lags < 1:
        raise ValueError(f"max_lags must be >=1, got {max_lags}")
    if min_lag < 1:
        raise ValueError(f"min_lag must be >=1, got {min_lag}")
    if not 0 < fraction_max <= 1:
        raise ValueError(f"fraction_max must be in (0,1], got {fraction_max}")


def _collect_tau_lengths(dataset: DatasetLike | Sequence[int]) -> list[int]:
    if isinstance(dataset, (Mapping, MutableMapping)) and not isinstance(
        dataset, (list, tuple)
    ):
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
        lengths = [int(length) for length in dataset]  # type: ignore[arg-type]

    if not lengths:
        raise ValueError("No split lengths available for tau derivation")
    if any(length <= 0 for length in lengths):
        raise ValueError(f"Non-positive split length encountered: {lengths}")
    return lengths


def _ensure_minimum_length(lengths: Sequence[int], *, min_lag: int) -> int:
    min_length = min(lengths)
    if min_length <= min_lag:
        raise ValueError(
            "Minimum split length "
            f"{min_length} is not greater than min_lag {min_lag}; cannot derive taus."
        )
    return min_length


def _derive_geometric_taus(
    *,
    min_length: int,
    min_lag: int,
    fraction_max: float,
    max_lags: int,
    base: Sequence[int] | None,
) -> list[int]:
    if base is not None:
        logger.warning(
            "derive_taus: 'base' provided but ignored because geometric=True"
        )

    upper_bound = int(max(min_lag + 1, np.floor(min_length * fraction_max)))
    upper_bound = min(upper_bound, min_length - 1)
    if upper_bound <= min_lag:
        raise ValueError(
            "Upper bound "
            f"{upper_bound} not greater than min_lag {min_lag}; cannot derive taus."
        )

    start = np.log(min_lag)
    stop = np.log(upper_bound)
    raw = np.exp(np.linspace(start, stop, num=max_lags))
    candidates = [int(round(value)) for value in raw]

    taus: list[int] = []
    last = 0
    for tau_candidate in candidates:
        if tau_candidate < min_lag or tau_candidate >= min_length:
            continue
        if tau_candidate <= last:
            continue
        taus.append(tau_candidate)
        last = tau_candidate

    if not taus:
        raise ValueError(
            "Geometric tau derivation yielded empty set "
            f"(min_length={min_length}, min_lag={min_lag}, upper={upper_bound})."
        )
    return taus


def _derive_base_taus(
    *,
    base: Sequence[int] | None,
    min_length: int,
    min_lag: int,
) -> list[int]:
    if base is None:
        raise ValueError("Non-geometric tau derivation requires a 'base' sequence")
    if not base:
        raise ValueError("Base tau candidate sequence is empty")

    invalid = [
        candidate
        for candidate in base
        if (not isinstance(candidate, (int, np.integer))) or int(candidate) <= 0
    ]
    if invalid:
        raise ValueError(
            "Base tau sequence must contain only positive integers, "
            f"got invalid entries {invalid}"
        )

    taus: list[int] = []
    seen: set[int] = set()
    for candidate in base:
        tau_value = int(candidate)
        if tau_value >= min_length or tau_value in seen or tau_value < min_lag:
            continue
        taus.append(tau_value)
        seen.add(tau_value)

    if not taus:
        raise ValueError(
            "Base tau filtering produced empty set "
            f"(base={list(base)}, min_length={min_length}, min_lag={min_lag})"
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
        logger.info(
            "Validated user taus %s (min split length %d)", taus_used, min_length
        )

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
