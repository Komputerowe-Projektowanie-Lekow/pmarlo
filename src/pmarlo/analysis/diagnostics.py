from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np
from sklearn.cross_decomposition import CCA

from pmarlo import constants as const

from .discretize import (
    _coerce_array,
    _normalise_splits,
    _resolve_shard_segments_for_split,
)
from .project_cv import apply_whitening_from_metadata

logger = logging.getLogger(__name__)

DatasetLike = Mapping[str, Any] | MutableMapping[str, Any]
_MAX_TAU_FRACTION = 1.0 / 3.0
_MIN_CK_MULTIPLIER = 2.0
_MAX_CK_MULTIPLIER = 5.0


@dataclass(frozen=True)
class _SegmentDescriptor:
    length: int
    stride: int


def _segment_descriptors_for_split(
    dataset: DatasetLike,
    split_name: str,
    split: Any,
    total_frames: int,
) -> list[_SegmentDescriptor]:
    lengths, strides = _resolve_shard_segments_for_split(
        dataset,
        split_name,
        split,
        total_frames,
    )
    descriptors: list[_SegmentDescriptor] = []
    for length, stride in zip(lengths, strides):
        descriptors.append(
            _SegmentDescriptor(length=int(length), stride=max(1, int(stride)))
        )
    consumed = sum(descriptor.length for descriptor in descriptors)
    if consumed != int(total_frames):
        raise ValueError(
            f"Segment metadata for split '{split_name}' spans {consumed} frames "
            f"but split contains {total_frames} frames"
        )
    if not descriptors:
        raise ValueError(f"No shard descriptors available for split '{split_name}'")
    return descriptors


def _prepare_tau_grid(taus: Sequence[int]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for value in sorted(int(t) for t in taus):
        if value < 1 or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return [0] + unique


def _integrated_autocorrelation_time(
    taus: Sequence[int],
    values: Sequence[float],
) -> float:
    if len(taus) != len(values) or not taus:
        return float("nan")
    tau_arr = np.asarray(taus, dtype=np.float64)
    rho_arr = np.asarray(values, dtype=np.float64)
    tau_int = 1.0
    last_tau = 0.0
    last_val = 1.0
    for tau, rho in zip(tau_arr[1:], rho_arr[1:]):
        if not np.isfinite(rho) or rho <= 0.0:
            break
        delta = tau - last_tau
        if delta <= 0.0:
            continue
        avg = 0.5 * (last_val + rho)
        tau_int += 2.0 * avg * delta
        last_tau = tau
        last_val = rho
    return float(max(1.0, tau_int))


def _recommend_ck_lags(
    tau_int: float,
    tau_limit: int,
) -> tuple[list[int], tuple[int, int] | None]:
    if not np.isfinite(tau_int) or tau_int <= 0.0:
        return [], None
    lower = max(1, int(np.ceil(_MIN_CK_MULTIPLIER * tau_int)))
    upper_bound = int(np.ceil(min(_MAX_CK_MULTIPLIER * tau_int, tau_limit)))
    if upper_bound < lower:
        upper_bound = lower
    if lower <= 0:
        lower = 1
    if upper_bound == lower:
        return [lower], (lower, upper_bound)
    span = upper_bound - lower
    num_points = min(5, span + 1)
    raw = np.geomspace(lower, upper_bound, num=num_points)
    candidates = sorted({max(lower, min(upper_bound, int(round(val)))) for val in raw})
    if candidates[0] > lower:
        candidates.insert(0, lower)
    if candidates[-1] < upper_bound:
        candidates.append(upper_bound)
    return candidates, (lower, upper_bound)


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


def _canonical_correlations(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    regularisation: float = const.NUMERIC_RELATIVE_TOLERANCE,
) -> list[float]:
    """Compute canonical correlations using scikit-learn's :class:`CCA`.

    Delegating to the library avoids maintaining a bespoke eigendecomposition
    routine and ensures we benefit from upstream numerical safeguards.  The
    returned canonical components are post-processed with ``numpy.corrcoef`` to
    obtain per-dimension Pearson correlation coefficients.

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
        x_comp = np.asarray(X_c[:, idx], dtype=np.float64)
        y_comp = np.asarray(Y_c[:, idx], dtype=np.float64)
        if x_comp.size == 0 or y_comp.size == 0:
            continue
        var_x = float(np.var(x_comp))
        var_y = float(np.var(y_comp))
        if var_x <= regularisation or var_y <= regularisation:
            correlations.append(0.0)
            continue
        corr_matrix = np.corrcoef(x_comp, y_comp)
        corr_value = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else 0.0
        if not np.isfinite(corr_value):
            corr_value = 0.0
        correlations.append(float(np.clip(abs(corr_value), 0.0, 1.0)))
    return correlations


# --- Autocorrelation helpers (refactored for SOLID) ------------------------------


def _validate_autocorr_input(X: np.ndarray) -> np.ndarray | None:
    """Validate and mean-center input for autocorrelation.

    Returns centered array or None if insufficient samples.
    Logs any issues instead of silent failure.
    """
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        logger.warning("Autocorrelation: expected 2D array, got ndim=%d", arr.ndim)
        return None
    if arr.shape[0] < 2:
        logger.error("Autocorrelation: insufficient samples (n=%d < 2)", arr.shape[0])
        return None
    if not np.isfinite(arr).all():
        logger.warning(
            "Autocorrelation: non-finite values detected; filtering may produce NaNs"
        )
    return arr


def _autocorrelation_curve(
    X: np.ndarray,
    taus: Sequence[int],
    segments: Sequence[_SegmentDescriptor],
) -> Dict[str, Any]:
    """Compute shard-wise autocorrelation and summary statistics."""

    Xc = _validate_autocorr_input(X)
    tau_grid = _prepare_tau_grid(taus)
    if Xc is None or not tau_grid:
        logger.warning(
            "Autocorrelation: invalid input; returning NaNs for %d taus", len(tau_grid)
        )
        nan_curve = [float("nan") for _ in tau_grid]
        return {
            "taus": [int(t) for t in tau_grid],
            "values": nan_curve,
            "tau_int": float("nan"),
            "lag_window": None,
            "recommended_ck_lags": [],
        }

    tau_array = np.asarray(tau_grid, dtype=np.int32)
    numerator = np.zeros_like(tau_array, dtype=np.float64)
    denominator = np.zeros_like(tau_array, dtype=np.float64)
    usable_lengths = [
        descriptor.length for descriptor in segments if descriptor.length > 1
    ]
    if usable_lengths:
        min_segment_length = min(usable_lengths)
    else:
        min_segment_length = min(descriptor.length for descriptor in segments)
    cursor = 0
    for descriptor in segments:
        seg_len = descriptor.length
        if seg_len <= 1:
            cursor += seg_len
            continue
        end = cursor + seg_len
        if end > Xc.shape[0]:
            raise ValueError(
                f"Segment descriptors exceed split length ({end} > {Xc.shape[0]})"
            )
        block = Xc[cursor:end]
        cursor = end
        centered = block - np.mean(block, axis=0, keepdims=True)
        variance = np.mean(centered * centered, axis=0)
        valid_mask = variance > const.NUMERIC_RELATIVE_TOLERANCE
        if not np.any(valid_mask):
            continue
        normalised = centered[:, valid_mask] / np.sqrt(variance[valid_mask])
        for idx, tau in enumerate(tau_array):
            if tau == 0:
                shard_value = 1.0
            elif tau >= seg_len:
                continue
            else:
                head = normalised[: seg_len - tau]
                tail = normalised[tau:]
                shard_value = float(np.mean(np.mean(head * tail, axis=0)))
            if not np.isfinite(shard_value):
                continue
            weight = seg_len - tau
            if weight <= 0:
                continue
            numerator[idx] += weight * shard_value
            denominator[idx] += weight
    if cursor != Xc.shape[0]:
        raise ValueError(
            f"Segment metadata consumed {cursor} frames but split contains {Xc.shape[0]}"
        )

    averaged = np.full_like(tau_array, fill_value=np.nan, dtype=np.float64)
    mask = denominator > 0.0
    averaged[mask] = numerator[mask] / denominator[mask]
    averaged = np.clip(averaged, -1.0, 1.0, out=averaged)
    if averaged.size > 0:
        averaged[0] = 1.0

    tau_list = [int(tau) for tau in tau_array]
    values_list = [float(val) if np.isfinite(val) else float("nan") for val in averaged]

    nan_indices = [i for i, val in enumerate(values_list) if not np.isfinite(val)]
    if nan_indices:
        logger.warning(
            "Autocorrelation: %d/%d NaN values in curve (first indices: %s)",
            len(nan_indices),
            len(values_list),
            nan_indices[:10],
        )

    tau_limit = max(
        1,
        int(np.floor(min_segment_length * _MAX_TAU_FRACTION)),
    )
    tau_int = _integrated_autocorrelation_time(tau_list, values_list)
    recommended, window = _recommend_ck_lags(tau_int, tau_limit)

    return {
        "taus": tau_list,
        "values": values_list,
        "tau_int": float(tau_int),
        "lag_window": window,
        "recommended_ck_lags": recommended,
    }


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
    max_lags: int = 10,
    min_lag: int = 1,
    fraction_max: float = _MAX_TAU_FRACTION,
    geometric: bool = True,
    base: Sequence[int] | None = None,
) -> list[int]:
    """Return a validated list of autocorrelation lag times for the dataset."""

    _validate_tau_parameters(
        max_lags=max_lags,
        min_lag=min_lag,
        fraction_max=fraction_max,
    )

    lengths, strides = _collect_tau_lengths(dataset)
    min_length = _ensure_minimum_length(lengths, min_lag=min_lag)
    stride_hint = min(strides) if strides else 1
    min_lag_effective = max(min_lag, stride_hint)

    if geometric:
        taus = _derive_geometric_taus(
            min_length=min_length,
            min_lag=min_lag_effective,
            fraction_max=fraction_max,
            max_lags=max_lags,
            base=base,
        )
        strategy = "geometric"
    else:
        taus = _derive_base_taus(
            base=base,
            min_length=min_length,
            min_lag=min_lag_effective,
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


def _collect_tau_lengths(
    dataset: DatasetLike | Sequence[int],
) -> tuple[list[int], list[int]]:
    if isinstance(dataset, (Mapping, MutableMapping)) and not isinstance(
        dataset, (list, tuple)
    ):
        splits = _normalise_splits(dataset)
        lengths: list[int] = []
        strides: list[int] = []
        for name, value in splits.items():
            try:
                arr = _coerce_array(value)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Skipping split during tau derivation: %s", exc)
                continue
            split_lengths, split_strides = _resolve_shard_segments_for_split(
                dataset,
                str(name),
                value,
                arr.shape[0],
            )
            lengths.extend(int(length) for length in split_lengths)
            strides.extend(max(1, int(stride)) for stride in split_strides)
    else:
        lengths = [int(length) for length in dataset]  # type: ignore[arg-type]
        strides = [1 for _ in lengths]

    if not lengths:
        raise ValueError("No split lengths available for tau derivation")
    if any(length <= 0 for length in lengths):
        raise ValueError(f"Non-positive split length encountered: {lengths}")
    return lengths, strides


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
        processed = _compute_split_diagnostics(name, split, taus_used, dataset)
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
    dataset: DatasetLike,
) -> tuple[list[float] | None, Dict[str, Any], list[str]] | None:
    """Gather canonical correlations and autocorrelation curve for one split.

    Raises errors if data is malformed. Returns None only if split cannot be coerced.
    """
    try:
        X = _coerce_array(split)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Skipping diagnostic split %s: %s", name, exc)
        return None

    metadata = split.get("meta") if isinstance(split, Mapping) else None
    if metadata is not None:
        whitened_full, _ = apply_whitening_from_metadata(X, metadata)
    else:
        whitened_full = X

    canonical: list[float] | None = None
    warnings: list[str] = []
    inputs = _extract_optional_inputs(split) if isinstance(split, Mapping) else None
    canonical_inputs = inputs
    canonical_values = whitened_full
    if inputs is not None:
        if inputs.shape[0] != whitened_full.shape[0]:
            length = min(inputs.shape[0], whitened_full.shape[0])
            logger.debug(
                "Truncating inputs/whitened for canonical correlation: %s length -> %d",
                name,
                length,
            )
            canonical_inputs = inputs[:length]
            canonical_values = whitened_full[:length]
        try:
            correlations = _canonical_correlations(canonical_inputs, canonical_values)
        except InsufficientSamplesError:
            logger.error(
                "%s: insufficient samples for canonical correlation (need >=2)", name
            )
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

    descriptors = _segment_descriptors_for_split(
        dataset,
        name,
        split,
        whitened_full.shape[0],
    )
    autocorr = _autocorrelation_curve(whitened_full, taus, descriptors)
    values = autocorr.get("values", [])
    if len(values) >= 4 and np.isfinite(values[1]) and np.isfinite(values[3]):
        if abs(values[1] - values[3]) < 0.05:
            msg = f"{name}: CV autocorrelation flat across early lags"
            warnings.append(msg)
            logger.warning(msg)

    return canonical, autocorr, warnings
