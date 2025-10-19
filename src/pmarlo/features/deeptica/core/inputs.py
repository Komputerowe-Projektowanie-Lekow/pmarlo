from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence, cast

import numpy as np
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from typing_extensions import Self

from .utils import set_all_seeds


class _StandardScalerProtocol(Protocol):
    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None: ...

    with_mean: bool
    with_std: bool
    mean_: np.ndarray | None
    scale_: np.ndarray | None

    def fit(self, X: np.ndarray) -> Self: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray: ...


StandardScaler: type[_StandardScalerProtocol]
StandardScaler = cast(type[_StandardScalerProtocol], SkStandardScaler)


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
