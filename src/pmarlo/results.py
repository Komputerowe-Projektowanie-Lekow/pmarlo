from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar

import numpy as np

logger = logging.getLogger("pmarlo")

T = TypeVar("T", bound="BaseResult")


@dataclass
class BaseResult:
    """Base class for result containers with serialization helpers."""

    version: ClassVar[str] = "1.0"

    def to_dict(self, metadata_only: bool = False) -> Dict[str, Any]:
        """Convert result to a serializable dict.

        If ``metadata_only`` is True, large arrays are reduced to shape and dtype
        information.
        """

        def _serialize(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                if metadata_only:
                    return {"shape": list(value.shape), "dtype": str(value.dtype)}
                return value.tolist()
            if isinstance(value, list):
                return [_serialize(v) for v in value]
            return value

        data = {k: _serialize(v) for k, v in asdict(self).items()}
        data["version"] = self.version
        return data

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Recreate a result object from a dict."""
        version = data.pop("version", None)
        if version != cls.version:
            logger.error(
                "Version mismatch when loading %s: %s != %s",
                cls.__name__,
                version,
                cls.version,
            )
            raise ValueError(f"Version mismatch: {version} != {cls.version}")

        def _deserialize(name: str, value: Any) -> Any:
            for f in fields(cls):
                if f.name == name:
                    if f.type == np.ndarray and isinstance(value, list):
                        return np.asarray(value)
                    return value
            return value

        kwargs = {k: _deserialize(k, v) for k, v in data.items()}
        return cls(**kwargs)  # type: ignore[arg-type]

    def to_json(self, metadata_only: bool = False) -> str:
        """Serialize result to a JSON string."""
        return json.dumps(self.to_dict(metadata_only=metadata_only))

    @classmethod
    def from_json(cls: Type[T], text: str) -> T:
        """Deserialize result from a JSON string."""
        data = json.loads(text)
        return cls.from_dict(data)

    def to_pickle(self, path: Path) -> None:
        """Serialize result to a pickle file."""
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def from_pickle(cls: Type[T], path: Path) -> T:
        """Load result object from a pickle file with version check."""
        with path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        if getattr(obj, "version", None) != cls.version:
            logger.error(
                "Version mismatch when loading %s: %s != %s",
                cls.__name__,
                getattr(obj, "version", None),
                cls.version,
            )
            raise ValueError(
                f"Version mismatch: {getattr(obj, 'version', None)} != {cls.version}"
            )
        return obj


@dataclass
class REMDResult(BaseResult):
    """Result container for REMD simulations."""

    temperatures: np.ndarray
    n_replicas: int
    exchange_frequency: int
    exchange_attempts: int
    exchanges_accepted: int
    final_acceptance_rate: float
    replica_states: List[int]
    state_replicas: List[int]
    exchange_history: List[List[int]]
    trajectory_files: List[str]
    acceptance_matrix: Optional[np.ndarray] = None
    replica_visitation_histogram: Optional[np.ndarray] = None
    frames_per_replica: List[int] = field(default_factory=list)
    effective_sample_size: Optional[float] = None


@dataclass
class DemuxResult(BaseResult):
    """Result of demultiplexing trajectories."""

    trajectory_file: str
    frames: int


@dataclass
class ClusteringResult(BaseResult):
    """Clustering assignments and centers."""

    assignments: np.ndarray
    centers: np.ndarray


@dataclass
class MSMResult(BaseResult):
    """Core MSM matrices and distributions."""

    transition_matrix: np.ndarray
    count_matrix: np.ndarray
    free_energies: Optional[np.ndarray] = None
    stationary_distribution: Optional[np.ndarray] = None

    @property
    def output_shape(self) -> tuple[int, ...]:
        """Number of states represented by the MSM."""
        return (self.transition_matrix.shape[0],)


@dataclass
class FESResult(BaseResult):
    """Free energy surface data and metadata."""

    free_energy: np.ndarray
    xedges: np.ndarray
    yedges: np.ndarray
    cv1_name: str
    cv2_name: str
    temperature: float

    @property
    def output_shape(self) -> tuple[int, int]:
        """Grid shape of the free energy surface."""
        return (int(self.free_energy.shape[0]), int(self.free_energy.shape[1]))


@dataclass
class CKResult(BaseResult):
    """Chapman-Kolmogorov test results."""

    lag_times: np.ndarray
    timescales: np.ndarray


@dataclass
class ITSResult(BaseResult):
    """Implied timescale estimation results with confidence intervals."""

    lag_times: np.ndarray
    eigenvalues: np.ndarray
    eigenvalues_ci: np.ndarray
    timescales: np.ndarray
    timescales_ci: np.ndarray
    rates: np.ndarray
    rates_ci: np.ndarray
    recommended_lag_window: Optional[tuple[float, float]] = None
