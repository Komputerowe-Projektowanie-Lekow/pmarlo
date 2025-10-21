"""Helpers for preparing FES inputs with consistent whitening."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
from scipy import ndimage

from pmarlo import constants as const

from .project_cv import apply_whitening_from_metadata

DatasetLike = MutableMapping[str, Any]


def _normalise_weights(
    n_frames: int, weights: np.ndarray | None
) -> tuple[np.ndarray, float, float]:
    if weights is None:
        if n_frames <= 0:
            raise ValueError("Cannot normalise weights with zero frames")
        w = np.full((n_frames,), 1.0 / float(n_frames), dtype=np.float64)
        return w, float(n_frames), float(n_frames)

    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.ndim != 1:
        raise ValueError("Frame weights must be one-dimensional")
    if w.shape[0] != n_frames:
        raise ValueError("Frame weights must match number of frames")
    if np.any(w < 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("Frame weights must be finite and non-negative")

    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("Frame weights must sum to a positive value")

    norm = w / total
    ess = total**2 / float(np.sum(w**2)) if np.sum(w**2) > 0 else total
    return norm, total, ess


def _resolve_kde_bins(bins: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(bins, int):
        count = bins
        if count < 2:
            raise ValueError("KDE FES requires at least two bins per dimension")
        return count, count
    elif isinstance(bins, Sequence):
        if len(bins) != 2:
            raise ValueError("KDE FES expects a sequence of two bin counts")
        x = int(bins[0])
        y = int(bins[1])
        if x < 2 or y < 2:
            raise ValueError("KDE FES requires at least two bins per dimension")
        return x, y
    else:
        # Handle other scalar types (like numpy scalars)
        try:
            count = int(bins)
            if count < 2:
                raise ValueError("KDE FES requires at least two bins per dimension")
            return count, count
        except (TypeError, ValueError) as exc:
            raise TypeError("Unsupported bins specification for KDE FES") from exc


def _compute_bandwidth(
    coord: np.ndarray,
    weights: np.ndarray,
    ess: float,
    selector: str | float,
) -> float:
    if isinstance(selector, (float, int)):
        value = float(selector)
        if value <= 0:
            raise ValueError("Bandwidth must be positive")
        return value

    selector_norm = str(selector).lower()
    mean = float(np.average(coord, weights=weights))
    var = float(np.average((coord - mean) ** 2, weights=weights))
    if var <= 0.0:
        raise ValueError("Coordinate variance must be positive to compute bandwidth")
    std = np.sqrt(var)
    d = 2.0
    n_eff = max(ess, 1.0)

    if selector_norm == "scott":
        factor = n_eff ** (-1.0 / (d + 4.0))
    elif selector_norm == "silverman":
        factor = (n_eff * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
    else:
        raise ValueError("Bandwidth must be 'scott', 'silverman', or a positive float")

    bandwidth = std * factor
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("Computed bandwidth must be finite and positive")
    return float(bandwidth)


def _compute_kde_surface(
    coord_x: np.ndarray,
    coord_y: np.ndarray,
    weights: np.ndarray | None,
    *,
    bins: int | Sequence[int],
    bandwidth: str | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_frames = coord_x.shape[0]
    w_norm, total_weight, ess = _normalise_weights(
        n_frames, None if weights is None else weights
    )

    nx, ny = _resolve_kde_bins(bins)
    bw_x = _compute_bandwidth(coord_x, w_norm, ess, bandwidth)
    bw_y = _compute_bandwidth(coord_y, w_norm, ess, bandwidth)

    pad_x = 3.0 * bw_x
    pad_y = 3.0 * bw_y
    x_min = float(np.min(coord_x)) - pad_x
    x_max = float(np.max(coord_x)) + pad_x
    y_min = float(np.min(coord_y)) - pad_y
    y_max = float(np.max(coord_y)) + pad_y

    if (
        not np.isfinite(x_min)
        or not np.isfinite(x_max)
        or x_min >= x_max
        or not np.isfinite(y_min)
        or not np.isfinite(y_max)
        or y_min >= y_max
    ):
        raise ValueError("Coordinate range must be finite and strictly increasing")

    xedges = np.linspace(x_min, x_max, num=nx + 1, dtype=np.float64)
    yedges = np.linspace(y_min, y_max, num=ny + 1, dtype=np.float64)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    diff_x = (xcenters[:, None] - coord_x[None, :]) / bw_x
    diff_y = (ycenters[:, None] - coord_y[None, :]) / bw_y

    np.square(diff_x, out=diff_x)
    np.square(diff_y, out=diff_y)
    np.multiply(diff_x, -0.5, out=diff_x)
    np.multiply(diff_y, -0.5, out=diff_y)
    np.exp(diff_x, out=diff_x)
    np.exp(diff_y, out=diff_y)

    density = np.einsum("ik,jk,k->ij", diff_x, diff_y, w_norm)
    normaliser = 1.0 / (2.0 * np.pi * bw_x * bw_y)
    density *= normaliser

    metadata = {
        "bandwidth": {
            "selector": bandwidth,
            "x": bw_x,
            "y": bw_y,
            "effective_sample_size": ess,
            "total_weight": total_weight,
        },
    }
    return density, xedges, yedges, metadata


def _compute_histogram_surface(
    coord_x: np.ndarray,
    coord_y: np.ndarray,
    weights: np.ndarray | None,
    *,
    bins: int | Sequence[int],
    min_count_per_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    hist, xedges, yedges = np.histogram2d(coord_x, coord_y, bins=bins, weights=weights)
    hist = hist.astype(np.float64, copy=False)
    raw_total = float(np.sum(hist))

    smoothed_bins = 0
    if min_count_per_bin > 0:
        hist, smoothed_bins = _smooth_sparse_bins(hist, min_count_per_bin)
        if smoothed_bins > 0 and raw_total > 0:
            current_total = float(np.sum(hist))
            if current_total > 0:
                hist *= raw_total / current_total

    metadata: dict[str, Any] = {
        "min_count_per_bin": min_count_per_bin,
        "smoothed_bins": smoothed_bins,
    }
    if smoothed_bins > 0:
        metadata["smoothing"] = "neighbor_average"
    return hist, xedges, yedges, metadata


def _smooth_sparse_bins(hist: np.ndarray, min_count: int) -> tuple[np.ndarray, int]:
    mask = hist < float(min_count)
    if not np.any(mask):
        return hist, 0

    def _neighbor_mean(values: np.ndarray) -> float:
        centre = values.size // 2
        total = np.sum(values) - values[centre]
        return total / float(values.size - 1)

    neighbor_mean = ndimage.generic_filter(
        hist,
        _neighbor_mean,
        size=3,
        mode="nearest",
    )

    smoothed = hist.copy()
    targets = np.maximum(neighbor_mean, float(min_count))
    update_mask = mask & (neighbor_mean > 0.0) & (targets > smoothed)
    smoothed[update_mask] = targets[update_mask]
    smoothed_bins = int(np.count_nonzero(update_mask))
    return smoothed, smoothed_bins


def ensure_fes_inputs_whitened(dataset: DatasetLike | Mapping[str, Any]) -> bool:
    """Apply whitening to the continuous CVs used for FES generation."""

    if not isinstance(dataset, (MutableMapping, dict)):
        raise TypeError("Dataset must be a mutable mapping to apply whitening")

    if "X" not in dataset:
        raise KeyError("Dataset is missing 'X' array required for whitening")

    X = dataset["X"]  # type: ignore[index]
    if X is None:
        raise ValueError("Dataset provides no coordinate array for whitening")

    artifacts = dataset.get("__artifacts__")  # type: ignore[assignment]
    if not isinstance(artifacts, Mapping):
        raise KeyError("Dataset artifacts are required for whitening metadata")

    summary = artifacts.get("mlcv_deeptica")
    if not isinstance(summary, (MutableMapping, dict)):
        raise KeyError(
            "Dataset artifacts must include 'mlcv_deeptica' whitening metadata"
        )

    whitened, applied = apply_whitening_from_metadata(
        np.asarray(X, dtype=np.float64), summary
    )
    if not applied:
        raise RuntimeError("Expected whitening metadata to be applied")

    dataset["X"] = whitened  # type: ignore[index]
    return True


def _select_split(
    dataset: Mapping[str, Any], split: str | None
) -> tuple[str, Mapping[str, Any]]:
    splits = dataset.get("splits")
    if isinstance(splits, Mapping) and splits:
        if split is not None and split in splits:
            return str(split), splits[split]  # type: ignore[index]
        if "train" in splits:
            return "train", splits["train"]  # type: ignore[index]
        first_key = next(iter(splits))
        return str(first_key), splits[first_key]  # type: ignore[index]
    raise ValueError("Dataset must provide a 'splits' mapping for FES computation")


def _coerce_array(obj: Mapping[str, Any], key: str) -> np.ndarray:
    arr = obj.get(key)
    if arr is None:
        raise ValueError(f"Split is missing '{key}' array")
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D CV array for '{key}', got shape {out.shape}")
    if out.shape[0] == 0 or out.shape[1] < 2:
        raise ValueError("FES computation requires at least two CV dimensions")
    return out


def compute_weighted_fes(
    dataset: DatasetLike | Mapping[str, Any],
    *,
    split: str | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
    bins: int | Sequence[int] = 64,
    temperature_K: float = 300.0,
    method: str = "kde",
    bandwidth: str | float = "scott",
    min_count_per_bin: int = 1,
    apply_whitening: bool = True,
) -> dict[str, Any]:
    """Compute a weighted free energy surface via KDE or grid histogram."""

    split_name, split_data, coords = _prepare_fes_coordinates(
        dataset,
        split,
        apply_whitening=apply_whitening,
    )
    weights_arr = _resolve_frame_weights(split_name, split_data, dataset, weights)
    if weights_arr is not None and weights_arr.shape[0] != coords.shape[0]:
        raise ValueError("Frame weights must match the number of frames in the split")
    metadata = _base_fes_metadata(split_name, temperature_K, method, weights_arr)

    hist, xedges, yedges, meta_updates = _compute_fes_surface(
        coords,
        method=metadata["method"],
        weights=weights_arr,
        bins=bins,
        bandwidth=bandwidth,
        min_count_per_bin=min_count_per_bin,
    )
    metadata.update(meta_updates)

    free_energy = _finalize_free_energy(hist, temperature_K)

    return {
        "histogram": hist,
        "xedges": xedges,
        "yedges": yedges,
        "free_energy": free_energy,
        "metadata": metadata,
    }


def _prepare_fes_coordinates(
    dataset: DatasetLike | Mapping[str, Any],
    split: str | None,
    *,
    apply_whitening: bool,
) -> tuple[str, Mapping[str, Any], np.ndarray]:
    """Return the split metadata and whitened CV coordinates."""

    if not isinstance(dataset, (MutableMapping, dict)):
        raise ValueError("Dataset must be a mapping with 'splits'")

    if apply_whitening:
        ensure_fes_inputs_whitened(dataset)

    split_name, split_data = _select_split(dataset, split)
    coords = _coerce_array(split_data, "X")
    return split_name, split_data, coords


def _resolve_frame_weights(
    split_name: str,
    split_data: Mapping[str, Any],
    dataset: DatasetLike | Mapping[str, Any],
    explicit_weights: Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    """Return per-frame weights from explicit input or dataset metadata."""

    if explicit_weights is not None:
        return np.asarray(explicit_weights, dtype=np.float64).reshape(-1)

    candidate = split_data.get("weights") if isinstance(split_data, Mapping) else None
    if candidate is None and isinstance(dataset, Mapping):
        fw = dataset.get("frame_weights")
        if isinstance(fw, Mapping):
            candidate = fw.get(split_name)

    if candidate is None:
        return None

    weights = np.asarray(candidate, dtype=np.float64).reshape(-1)
    return weights


def _base_fes_metadata(
    split_name: str,
    temperature_K: float,
    method: str,
    weights: np.ndarray | None,
) -> dict[str, Any]:
    """Build the base metadata dictionary for FES results."""

    method_norm = str(method or "kde").lower()
    if method_norm not in {"kde", "grid"}:
        raise ValueError("FES method must be either 'kde' or 'grid'")

    return {
        "temperature_K": float(temperature_K),
        "split": split_name,
        "weighted": weights is not None,
        "method": method_norm,
    }


def _compute_fes_surface(
    coords: np.ndarray,
    *,
    method: str,
    weights: np.ndarray | None,
    bins: int | Sequence[int],
    bandwidth: str | float,
    min_count_per_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate KDE or histogram surfaces for the provided coordinates."""

    coord_x = coords[:, 0]
    coord_y = coords[:, 1]

    if method == "kde":
        surface, xedges, yedges, extra_meta = _compute_kde_surface(
            coord_x,
            coord_y,
            weights,
            bins=bins,
            bandwidth=bandwidth,
        )
        return surface, xedges, yedges, extra_meta

    hist, xedges, yedges, extra_meta = _compute_histogram_surface(
        coord_x,
        coord_y,
        weights,
        bins=bins,
        min_count_per_bin=int(min_count_per_bin),
    )
    return hist, xedges, yedges, extra_meta


def _finalize_free_energy(hist: np.ndarray, temperature_K: float) -> np.ndarray:
    """Convert histogram counts into a free-energy landscape."""

    if not np.all(np.isfinite(hist)):
        raise ValueError("Histogram must contain only finite values")

    total = float(np.sum(hist))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Histogram total must be positive and finite")

    if np.any(hist <= 0):
        raise ValueError("Histogram entries must be strictly positive for FES")

    prob = hist / total
    free_energy = -(
        const.BOLTZMANN_CONSTANT_KJ_PER_MOL * float(temperature_K)
    ) * np.log(prob)

    if not np.all(np.isfinite(free_energy)):
        raise FloatingPointError("Free energy computation produced non-finite values")

    free_energy = free_energy - np.min(free_energy)

    return free_energy
