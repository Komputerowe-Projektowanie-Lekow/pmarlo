"""Typed dataset containers for analysis routines."""

from __future__ import annotations

from collections.abc import Mapping as ABCMapping
from typing import Any, Dict, Mapping, MutableMapping, Sequence

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)


def _looks_like_split(value: Any) -> bool:
    if isinstance(value, (Mapping, MutableMapping)):
        candidate = value.get("X")
    elif hasattr(value, "X"):
        candidate = getattr(value, "X")
    else:  # pragma: no cover - defensive guardrail
        candidate = value

    try:
        arr = np.asarray(candidate)
    except Exception:
        return False

    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return False
    return bool(np.isfinite(arr).all())


class SplitData(BaseModel):
    """Validated representation of a single analysis split."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    X: np.ndarray
    weights: np.ndarray | None = None

    @field_validator("X", mode="before")
    @classmethod
    def _coerce_X(cls, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Split X must be 2D, got shape {arr.shape}")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("Split X must contain at least one row and column")
        if not np.isfinite(arr).all():
            raise ValueError("Split X must contain only finite values")
        return arr

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weights(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            raise ValueError("Split weights must not be empty")
        return arr

    @model_validator(mode="after")
    def _validate_weights(self) -> "SplitData":
        if self.weights is not None:
            if self.weights.shape[0] != self.X.shape[0]:
                raise ValueError(
                    "Split weights length does not match number of frames "
                    f"({self.weights.shape[0]} vs {self.X.shape[0]})"
                )
            if not np.isfinite(self.weights).all():
                raise ValueError("Split weights must contain only finite values")
        return self

    def to_mapping(self) -> Dict[str, Any]:
        data = self.model_dump()
        if self.model_extra:
            data.update(self.model_extra)
        return data


class AnalysisDataset(BaseModel, ABCMapping):
    """Validated dataset structure shared across analysis routines."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        populate_by_name=True,
    )

    splits: Dict[str, SplitData]
    X: np.ndarray | None = None
    frame_weights: Dict[str, np.ndarray] | None = None
    artifacts: Mapping[str, Any] | None = Field(default=None, alias="__artifacts__")
    dtrajs: Sequence[Sequence[int]] | None = None

    _source: Mapping[str, Any] | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _coerce_root(cls, value: Any) -> Mapping[str, Any]:
        if isinstance(value, AnalysisDataset):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("AnalysisDataset requires a mapping with split data")

        data = dict(value)
        raw_splits = data.get("splits")
        if raw_splits is None:
            candidates: Dict[str, Any] = {}
            for key, candidate in data.items():
                if str(key).startswith("__"):
                    continue
                if _looks_like_split(candidate):
                    candidates[str(key)] = candidate
            if not candidates:
                raise ValueError("AnalysisDataset requires a 'splits' mapping")
            data["splits"] = candidates
        elif not isinstance(raw_splits, Mapping):
            raise TypeError("'splits' must map split names to split definitions")
        return data

    @field_validator("splits", mode="before")
    @classmethod
    def _prepare_splits(cls, value: Any) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("'splits' must be a mapping")
        return {str(key): val for key, val in value.items()}

    @field_validator("X", mode="before")
    @classmethod
    def _coerce_dataset_matrix(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Dataset X must be 2D, got shape {arr.shape}")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("Dataset X must contain at least one row and column")
        if not np.isfinite(arr).all():
            raise ValueError("Dataset X must contain only finite values")
        return arr

    @field_validator("frame_weights", mode="before")
    @classmethod
    def _coerce_frame_weights(cls, value: Any) -> Dict[str, np.ndarray] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("frame_weights must be a mapping of split keys to arrays")
        prepared: Dict[str, np.ndarray] = {}
        for key, candidate in value.items():
            arr = np.asarray(candidate, dtype=np.float64).reshape(-1)
            if not np.isfinite(arr).all():
                raise ValueError(f"Frame weights for split '{key}' must be finite")
            prepared[str(key)] = arr
        return prepared

    @model_validator(mode="after")
    def _validate_frame_weights(self) -> "AnalysisDataset":
        if self.frame_weights:
            for name, weights in self.frame_weights.items():
                split = self.splits.get(name)
                if split is not None and weights.shape[0] != split.X.shape[0]:
                    raise ValueError(
                        "Frame weights length does not match split frame count "
                        f"({weights.shape[0]} vs {split.X.shape[0]}) for split '{name}'"
                    )
        return self

    @classmethod
    def from_like(cls, dataset: "DatasetInput") -> "AnalysisDataset":
        model = cls.model_validate(dataset)
        if isinstance(dataset, Mapping):
            model._source = dataset
        return model

    def splits_as_mappings(self) -> Dict[str, Dict[str, Any]]:
        return {name: split.to_mapping() for name, split in self.splits.items()}

    def as_mapping(self) -> Dict[str, Any]:
        data = self.model_dump(by_alias=True)
        data["splits"] = self.splits_as_mappings()
        return data

    # Mapping interface -----------------------------------------------------
    def __iter__(self):
        return iter(self.as_mapping())

    def __len__(self) -> int:
        return len(self.as_mapping())

    def __getitem__(self, key: str) -> Any:
        return self.as_mapping()[key]

    def get(self, key: str, default: Any = None) -> Any:  # pragma: no cover - helper
        return self.as_mapping().get(key, default)


DatasetInput = AnalysisDataset | Mapping[str, Any] | MutableMapping[str, Any]


def ensure_analysis_dataset(dataset: DatasetInput) -> AnalysisDataset:
    return AnalysisDataset.from_like(dataset)


__all__ = [
    "AnalysisDataset",
    "DatasetInput",
    "SplitData",
    "ensure_analysis_dataset",
    "_looks_like_split",
]
