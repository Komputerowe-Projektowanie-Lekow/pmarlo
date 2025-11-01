import json
import math
import re
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np

from pmarlo.utils.path_utils import ensure_directory

_UNSAFE_SLUG_CHARS = re.compile(r"[^a-z0-9_-]")

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


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _slugify(label: Optional[str]) -> Optional[str]:
    """Return a deterministic slug for ``label`` suitable for filenames."""
    if not label:
        return None
    normalised = unicodedata.normalize("NFKD", str(label)).strip()
    ascii_label = normalised.encode("ascii", "ignore").decode("ascii")
    slug = _UNSAFE_SLUG_CHARS.sub("_", ascii_label.lower())
    slug = slug.strip("_")
    return slug or None


def _coerce_path_list(paths: Iterable[str | Path]) -> List[Path]:
    return [Path(p).resolve() for p in paths]


def choose_sim_seed(mode: str, *, fixed: Optional[int] = None) -> Optional[int]:
    """Choose simulation seed based on mode."""
    import random

    if mode == "none":
        return None
    elif mode == "fixed":
        return fixed
    elif mode == "auto":
        return random.randint(1, 1000000)
    else:
        raise ValueError(f"Unknown seed mode: {mode}")


def _normalize_training_metrics(
    metrics: Mapping[str, Any] | None,
    *,
    tau_schedule: Optional[Sequence[Any]] = None,
    epochs_per_tau: Optional[int] = None,
) -> Dict[str, Any]:
    """Ensure Deep-TICA metrics expose best score/epoch/tau values."""

    if not isinstance(metrics, Mapping):
        return {}

    normalized: Dict[str, Any] = dict(metrics)

    raw_curve = normalized.get("val_score_curve")
    finite_scores: List[tuple[int, float]] = []
    if isinstance(raw_curve, Sequence):
        for idx, value in enumerate(raw_curve):
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                finite_scores.append((idx, score))

    best_val_score = normalized.get("best_val_score")
    if best_val_score is None and finite_scores:
        best_idx, best_score = max(finite_scores, key=lambda item: item[1])
        normalized["best_val_score"] = float(best_score)
        normalized.setdefault("_best_epoch_index", best_idx)
    elif finite_scores and isinstance(normalized.get("best_epoch"), (int, float)):
        idx = int(normalized["best_epoch"]) - 1
        if 0 <= idx < len(finite_scores):
            normalized.setdefault("_best_epoch_index", idx)

    best_epoch = normalized.get("best_epoch")
    if best_epoch is None and finite_scores:
        best_idx = normalized.get("_best_epoch_index")
        if not isinstance(best_idx, int):
            best_idx = max(finite_scores, key=lambda item: item[1])[0]
        normalized["best_epoch"] = int(best_idx + 1)
        if normalized.get("best_val_score") is None:
            normalized["best_val_score"] = float(finite_scores[best_idx][1])
    elif isinstance(best_epoch, (int, float)):
        normalized["best_epoch"] = int(best_epoch)

    if normalized.get("best_val_score") is not None:
        try:
            normalized["best_val_score"] = float(normalized["best_val_score"])
        except (TypeError, ValueError):
            normalized["best_val_score"] = None

    best_tau = normalized.get("best_tau")
    if best_tau is None:
        schedule: List[int] = []
        if isinstance(tau_schedule, Sequence):
            for item in tau_schedule:
                try:
                    schedule.append(int(item))
                except (TypeError, ValueError):
                    continue
        epochs = None
        if isinstance(epochs_per_tau, (int, float)):
            epochs = int(epochs_per_tau)
        if schedule and epochs and epochs > 0:
            idx = normalized.get("_best_epoch_index")
            if not isinstance(idx, int):
                if finite_scores:
                    idx = max(finite_scores, key=lambda item: item[1])[0]
                else:
                    idx = None
            if isinstance(idx, int):
                stage = max(0, min(idx // epochs, len(schedule) - 1))
                normalized["best_tau"] = schedule[stage]
    else:
        try:
            normalized["best_tau"] = int(best_tau)
        except (TypeError, ValueError):
            normalized["best_tau"] = None

    normalized.pop("_best_epoch_index", None)
    return normalized


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
