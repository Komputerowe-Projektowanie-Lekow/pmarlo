from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Self, Sequence, cast

import numpy as np

from .utils import set_all_seeds

try:  # pragma: no cover - optional ML stack
    from sklearn.preprocessing import StandardScaler as SkStandardScaler  # type: ignore
except Exception:  # pragma: no cover - sklearn optional dependency
    SkStandardScaler = None  # type: ignore[assignment]


class _StandardScalerProtocol(Protocol):
    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None: ...

    with_mean: bool
    with_std: bool
    mean_: np.ndarray | None
    scale_: np.ndarray | None

    def fit(self, X: np.ndarray) -> Self: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...


class _FallbackStandardScaler:  # type: ignore[too-many-instance-attributes]
    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> Self:
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Expected 2-D array for scaling")
        if self.with_mean:
            self.mean_ = np.mean(data, axis=0)
        else:
            self.mean_ = np.zeros(data.shape[1], dtype=np.float64)
        if self.with_std:
            scale = np.std(data, axis=0, ddof=0)
            scale[scale == 0.0] = 1.0
            self.scale_ = scale
        else:
            self.scale_ = np.ones(data.shape[1], dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler fallback requires fit before transform")
        data = np.asarray(X, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("Expected 2-D array for scaling")
        if self.with_mean:
            data = data - self.mean_
        if self.with_std:
            data = data / self.scale_
        return data

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


StandardScaler: type[_StandardScalerProtocol]
if SkStandardScaler is not None:
    StandardScaler = cast(type[_StandardScalerProtocol], SkStandardScaler)
else:
    StandardScaler = cast(type[_StandardScalerProtocol], _FallbackStandardScaler)


@dataclass(slots=True)
class FeaturePrep:
    """Prepared feature bundle used as input to the DeepTICA trainer."""

    X: np.ndarray
    Z: np.ndarray
    scaler: _StandardScalerProtocol
    tau_schedule: tuple[int, ...]
    input_dim: int
    seed: int


def prepare_features(
    arrays: Iterable[np.ndarray],
    *,
    tau_schedule: Sequence[int],
    seed: int,
) -> FeaturePrep:
    """Concatenate feature arrays, fit a scaler, and return float32 tensors."""

    set_all_seeds(seed)

    stacked = [np.asarray(block, dtype=np.float32) for block in arrays]
    if not stacked:
        raise ValueError("Expected at least one trajectory array for DeepTICA")

    X = np.concatenate(stacked, axis=0)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(
        np.asarray(X, dtype=np.float64)
    )
    Z = scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32, copy=False)

    schedule = tuple(int(t) for t in tau_schedule if int(t) > 0)
    if not schedule:
        raise ValueError("Tau schedule must contain at least one positive lag")

    return FeaturePrep(
        X=X,
        Z=Z,
        scaler=scaler,
        tau_schedule=schedule,
        input_dim=int(Z.shape[1]),
        seed=int(seed),
    )
