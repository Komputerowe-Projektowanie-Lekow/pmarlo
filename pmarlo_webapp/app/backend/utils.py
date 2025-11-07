import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, cast

import numpy as np

from pmarlo.api import choose_sim_seed, coerce_path_list, slugify, timestamp
from pmarlo.utils.path_utils import ensure_directory

# Re-export for backward compatibility within webapp
_timestamp = timestamp
_slugify = slugify
_coerce_path_list = coerce_path_list

_STRUCTURE_EXTENSIONS: tuple[str, ...] = (
    ".dcd",
    ".xtc",
    ".trr",
    ".nc",
    ".h5",
    ".hdf5",
    ".pdb",
    ".gro",
)


def _build_result_cls() -> "_BuildResult":
    return cast("_BuildResult", _pmarlo_handles()["BuildResult"])


def _sanitize_artifacts(data: Any) -> Any:
    return _pmarlo_handles()["_sanitize_artifacts"](data)


def _resolve_workspace_path(base: Path, candidate: Path) -> Path:
    if candidate.is_absolute():
        return candidate.expanduser().resolve()
    return (base / candidate).expanduser().resolve()



def _is_transition_matrix_reversible(
    T: np.ndarray, pi: np.ndarray, atol: float = 1e-8, rtol: float = 1e-5
) -> bool:
    """Check detailed balance condition for a transition matrix."""

    if T.size == 0 or pi.size == 0:
        return False
    flux = np.multiply(pi[:, None], T)
    return np.allclose(flux, flux.T, atol=atol, rtol=rtol)


def _load_projection_matrix(path: Path) -> np.ndarray:
    """Load DeepTICA projection matrix from .npz or .npy file."""
    if not path.exists():
        raise FileNotFoundError(f"DeepTICA projection file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            for key in ("projection", "deeptica", "X"):
                if key in data:
                    matrix = np.asarray(data[key], dtype=float)
                    break
            else:
                raise ValueError(
                    "DeepTICA projection archive must contain a 'projection' or 'deeptica' array"
                )
    elif suffix in {".npy"}:
        matrix = np.asarray(np.load(path), dtype=float)
    else:
        raise ValueError(
            f"Unsupported DeepTICA projection format '{suffix}'. Use .npz or .npy."
        )

    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("DeepTICA projection must be a non-empty 2D array")
    return matrix


def _load_metadata_mapping(path: Path) -> Dict[str, Any]:
    """Load DeepTICA metadata from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"DeepTICA metadata file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("DeepTICA metadata must decode to a mapping")
    return dict(payload)


@lru_cache(maxsize=1)
def _pmarlo_handles() -> Dict[str, Any]:
    """Import heavyweight PMARLO helpers on demand."""
    from pmarlo.api.shards import (
        build_from_shards as _build_from_shards,
        emit_shards_rg_rmsd_windowed as _emit_shards,
    )
    from pmarlo.api.replica_exchange import (
        run_replica_exchange as _run_replica_exchange,
    )
    from pmarlo.data.shard import read_shard as _read_shard
    from pmarlo.transform.build import BuildResult as _BuildResultRuntime
    from pmarlo.transform.build import _sanitize_artifacts as _sanitize

    return {
        "build_from_shards": _build_from_shards,
        "emit_shards_rg_rmsd_windowed": _emit_shards,
        "run_replica_exchange": _run_replica_exchange,
        "read_shard": _read_shard,
        "BuildResult": _BuildResultRuntime,
        "_sanitize_artifacts": _sanitize,
    }


def build_from_shards(*args: Any, **kwargs: Any) -> Any:
    return _pmarlo_handles()["build_from_shards"](*args, **kwargs)


def emit_shards_rg_rmsd_windowed(*args: Any, **kwargs: Any) -> Any:
    return _pmarlo_handles()["emit_shards_rg_rmsd_windowed"](*args, **kwargs)


def run_replica_exchange(*args: Any, **kwargs: Any) -> Any:
    return _pmarlo_handles()["run_replica_exchange"](*args, **kwargs)


def read_shard(*args: Any, **kwargs: Any) -> Any:
    return _pmarlo_handles()["read_shard"](*args, **kwargs)

# Module-level constants
_STRUCTURE_EXTENSIONS = {".pdb", ".gro", ".cif", ".dcd", ".xtc", ".trr", ".nc"}
