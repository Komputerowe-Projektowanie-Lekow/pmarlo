from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler  # type: ignore

from .utils import set_all_seeds


@dataclass(slots=True)
class FeaturePrep:
    """Prepared feature bundle used as input to the DeepTICA trainer."""

    X: np.ndarray
    Z: np.ndarray
    scaler: StandardScaler
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
    scaler = StandardScaler(with_mean=True, with_std=True).fit(np.asarray(X, dtype=np.float64))
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
