from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np

from .discretize import _coerce_array, _normalise_splits
from .project_cv import apply_whitening_from_metadata

logger = logging.getLogger(__name__)

DatasetLike = Mapping[str, Any] | MutableMapping[str, Any]
_TAU_SEQUENCE: tuple[int, ...] = (2, 5, 10, 20, 40)


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


def _canonical_correlations(
    X: np.ndarray, Y: np.ndarray, *, regularisation: float = 1e-8
) -> list[float]:
    n = min(X.shape[0], Y.shape[0])
    if n < 2:
        return []
    X = X[:n]
    Y = Y[:n]
    Xc = X - np.mean(X, axis=0, keepdims=True)
    Yc = Y - np.mean(Y, axis=0, keepdims=True)
    Sxx = (Xc.T @ Xc) / max(n - 1, 1)
    Syy = (Yc.T @ Yc) / max(n - 1, 1)
    Sxy = (Xc.T @ Yc) / max(n - 1, 1)

    def _inv_sqrt(mat: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(mat + regularisation * np.eye(mat.shape[0]))
        eigvals = np.clip(eigvals, a_min=regularisation, a_max=None)
        inv_root = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return inv_root

    try:
        inv_sqrt_x = _inv_sqrt(Sxx)
        inv_sqrt_y = _inv_sqrt(Syy)
    except np.linalg.LinAlgError:
        return []
    M = inv_sqrt_x @ Sxy @ inv_sqrt_y
    try:
        _, s, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return []
    s = np.clip(s, 0.0, 1.0)
    return [float(val) for val in s]


def _autocorrelation_curve(X: np.ndarray, taus: Sequence[int]) -> list[float]:
    if X.shape[0] < 2:
        return [float("nan") for _ in taus]
    Xc = X - np.mean(X, axis=0, keepdims=True)
    curve: list[float] = []
    for tau in taus:
        if tau <= 0 or tau >= Xc.shape[0]:
            curve.append(float("nan"))
            continue
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
            curve.append(float("nan"))
        else:
            finite = rho[np.isfinite(rho)]
            value = float(np.mean(finite)) if finite.size else float("nan")
            curve.append(value)
    return curve


def compute_diagnostics(
    dataset: DatasetLike,
    *,
    diag_mass: float | None = None,
    taus: Sequence[int] = _TAU_SEQUENCE,
) -> Dict[str, Any]:
    """Compute triviality/stability diagnostics for downstream reporting."""

    splits = _normalise_splits(dataset)
    canonical: Dict[str, list[float]] = {}
    autocorr: Dict[str, Dict[str, Any]] = {}
    warnings: list[str] = []

    for name, split in splits.items():
        try:
            X = _coerce_array(split)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Skipping diagnostic split %s: %s", name, exc)
            continue
        metadata = None
        if isinstance(split, Mapping):
            metadata = split.get("meta")
        whitened, _ = apply_whitening_from_metadata(X, metadata)

        inputs = None
        if isinstance(split, Mapping):
            inputs = _extract_optional_inputs(split)
        if inputs is not None:
            if inputs.shape[0] != whitened.shape[0]:
                length = min(inputs.shape[0], whitened.shape[0])
                inputs = inputs[:length]
                whitened = whitened[:length]
            corr = _canonical_correlations(inputs, whitened)
            if corr:
                canonical[name] = corr
                if corr and min(corr) > 0.95:
                    msg = f"{name}: CVs reparametrize inputs"
                    warnings.append(msg)
                    logger.warning(msg)
        curve = _autocorrelation_curve(whitened, taus)
        autocorr[name] = {"taus": list(taus), "values": curve}
        if len(curve) >= 4 and np.isfinite(curve[0]) and np.isfinite(curve[3]):
            if abs(curve[0] - curve[3]) < 0.05:
                msg = f"{name}: CV autocorrelation flat across lags"
                warnings.append(msg)
                logger.warning(msg)

    if diag_mass is not None and np.isfinite(diag_mass) and diag_mass > 0.95:
        msg = f"MSM diagonal mass high ({diag_mass:.3f})"
        warnings.append(msg)
        logger.warning(msg)

    return {
        "canonical_correlation": canonical,
        "autocorrelation": autocorr,
        "diag_mass": float(diag_mass) if diag_mass is not None else None,
        "taus": list(taus),
        "warnings": warnings,
    }
