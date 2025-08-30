from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..progress import ProgressCB
from ..states import msm_bridge
from ..transform.apply import apply_transform_plan
from ..transform.plan import TransformPlan, TransformStep
from ..transform.runner import apply_plan as _apply_plan


@dataclass(frozen=True)
class BuildOpts:
    """
    Declarative build options for constructing analyses.

    Only declarative knobs belong here; decisions/choices resolved by search
    or heuristics should be represented in AppliedOpts instead.
    """

    seed: Optional[int] = None
    device: Optional[str] = None
    # Common knobs discovered in the codebase
    max_bins: Optional[int] = None
    lag_candidates: Optional[List[int]] = None
    count_mode: str = "sliding"  # or "strided"
    n_states: Optional[int] = None
    temperature: Optional[float] = None


@dataclass(frozen=True)
class AppliedOpts:
    """
    Concrete, resolved options applied for a particular build.

    This tracks provenance of the optimization/selection results.
    """

    bins: Optional[Dict[str, int]] = None
    lag: Optional[int] = None
    macrostates: Optional[int] = None
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunMetadata:
    """Provenance metadata attached to all build results."""

    transform_plan: Tuple[TransformStep, ...]
    applied_opts: AppliedOpts
    build_opts: BuildOpts
    dataset_hash: str
    digest: str
    schema_version: str = "1"
    # Structured, estimator-specific provenance
    fes: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        def _step_to_dict(step: TransformStep) -> Dict[str, Any]:
            # Sort param keys for stable serialization
            params = {k: step.params[k] for k in sorted(step.params.keys())}
            return {"name": step.name, "params": params}

        return {
            "transform_plan": [_step_to_dict(s) for s in self.transform_plan],
            "applied_opts": asdict(self.applied_opts),
            "build_opts": asdict(self.build_opts),
            "dataset_hash": self.dataset_hash,
            "digest": self.digest,
            "schema_version": self.schema_version,
            "fes": self.fes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunMetadata":
        steps = tuple(
            TransformStep(d["name"], dict(d.get("params", {})))
            for d in data.get("transform_plan", [])
        )
        applied = AppliedOpts(**data.get("applied_opts", {}))
        opts = BuildOpts(**data.get("build_opts", {}))
        dataset_hash = str(data.get("dataset_hash", ""))
        digest = str(data.get("digest", ""))
        schema_version = str(data.get("schema_version", "1"))
        return cls(
            transform_plan=steps,
            applied_opts=applied,
            build_opts=opts,
            dataset_hash=dataset_hash,
            digest=digest,
            schema_version=schema_version,
            fes=data.get("fes"),
        )


@dataclass
class BuildResult:
    """
    Unified result envelope exposing provenance via .metadata and optional
    sub-results like MSM or FES.
    """

    metadata: RunMetadata
    # Minimal MSM payload for provenance-aware downstream usage
    transition_matrix: Optional[np.ndarray] = None
    stationary_distribution: Optional[np.ndarray] = None
    # Optional payloads from other estimators
    fes: Any | None = None
    tram: Any | None = None
    flags: Dict[str, bool] = field(default_factory=dict)
    messages: List[str] = field(default_factory=list)
    artifacts: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        """Serialize BuildResult using base64-encoded arrays for portability."""

        def _encode_array(arr: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
            if arr is None:
                return None
            a = np.asarray(arr)
            payload = base64.b64encode(a.tobytes()).decode("ascii")
            return {
                "dtype": str(a.dtype),
                "shape": list(a.shape),
                "data": payload,
                "byteorder": a.dtype.byteorder,
            }

        def _serialize_generic(value: Any) -> Any:
            import dataclasses as _dc

            if isinstance(value, np.ndarray):
                return _encode_array(value)
            # Only treat dataclass instances (not dataclass classes) as serializable
            if _dc.is_dataclass(value) and not isinstance(value, type):
                return {k: _serialize_generic(v) for k, v in _dc.asdict(value).items()}
            if isinstance(value, (list, tuple)):
                return [_serialize_generic(v) for v in value]
            if isinstance(value, dict):
                return {k: _serialize_generic(v) for k, v in value.items()}
            return value

        obj = {
            "metadata": self.metadata.to_dict(),
            "transition_matrix": _encode_array(self.transition_matrix),
            "stationary_distribution": _encode_array(self.stationary_distribution),
            "fes": _serialize_generic(self.fes),
            "tram": _serialize_generic(self.tram),
            "artifacts": _serialize_generic(self.artifacts),
        }
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "BuildResult":
        """Deserialize BuildResult from JSON produced by to_json."""
        from pmarlo.fes.surfaces import FESResult  # local import to avoid cycles

        data = json.loads(text)
        md = RunMetadata.from_dict(data["metadata"]) if "metadata" in data else None

        def _decode_array(obj: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
            if obj is None:
                return None
            dtype = np.dtype(str(obj["dtype"]))
            shape = tuple(int(x) for x in obj["shape"])
            raw = base64.b64decode(obj["data"])  # type: ignore[arg-type]
            arr = np.frombuffer(raw, dtype=dtype)
            out = arr.reshape(shape)
            assert out.dtype == dtype
            return out

        def _decode_generic(value: Any) -> Any:
            if isinstance(value, dict):
                if {"dtype", "shape", "data"}.issubset(value.keys()):
                    return _decode_array(value)  # type: ignore[arg-type]
                return {k: _decode_generic(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_decode_generic(v) for v in value]
            return value

        fes = None
        if isinstance(data.get("fes"), dict):
            d = data["fes"]
            if {"F", "xedges", "yedges"}.issubset(d):
                fes = FESResult(
                    F=_decode_array(d.get("F")),  # type: ignore[arg-type]
                    xedges=_decode_array(d.get("xedges")),  # type: ignore[arg-type]
                    yedges=_decode_array(d.get("yedges")),  # type: ignore[arg-type]
                    levels_kJmol=_decode_array(d.get("levels_kJmol")),  # type: ignore[arg-type]
                    metadata=_decode_generic(d.get("metadata", {})),
                )
        artifacts = (
            _decode_generic(data.get("artifacts")) if "artifacts" in data else None
        )
        return cls(
            metadata=(
                md
                if md is not None
                else RunMetadata(
                    (), AppliedOpts(), BuildOpts(), dataset_hash="", digest=""
                )
            ),
            transition_matrix=_decode_array(data.get("transition_matrix")),
            stationary_distribution=_decode_array(data.get("stationary_distribution")),
            fes=fes,
            tram=_decode_generic(data.get("tram")),
            artifacts=artifacts,
        )


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        np.random.seed(seed)
    except Exception:
        # Best-effort; seed is optional
        pass


def _extract_dtrajs(dataset: Any) -> List[np.ndarray]:
    """
    Extract discrete trajectories from various dataset shapes.

    Supports:
    - {"dtrajs": List[np.ndarray]}
    - objects with attribute .dtrajs
    - a single np.ndarray interpreted as one dtraj
    """
    if isinstance(dataset, dict) and "dtrajs" in dataset:
        return [np.asarray(dt, dtype=int) for dt in dataset["dtrajs"]]
    if hasattr(dataset, "dtrajs"):
        return [np.asarray(dt, dtype=int) for dt in getattr(dataset, "dtrajs")]
    if isinstance(dataset, np.ndarray):
        return [np.asarray(dataset, dtype=int)]
    return []


def _extract_cvs_from_dict_explicit(
    dataset: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[str, str], Tuple[bool, bool]]]:
    if "cv1" in dataset and "cv2" in dataset:
        names = tuple(dataset.get("cv_names", ("cv1", "cv2")))  # type: ignore[assignment]
        names = names if len(names) == 2 else ("cv1", "cv2")
        periodic = tuple(dataset.get("periodic", (False, False)))  # type: ignore[assignment]
        periodic = periodic if len(periodic) == 2 else (False, False)
        return (
            np.asarray(dataset["cv1"], dtype=float),
            np.asarray(dataset["cv2"], dtype=float),
            (str(names[0]), str(names[1])),
            (bool(periodic[0]), bool(periodic[1])),
        )
    return None


def _extract_cvs_from_dict_map(
    dataset: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[str, str], Tuple[bool, bool]]]:
    cvs_map = dataset.get("cvs")
    if isinstance(cvs_map, dict) and len(cvs_map) >= 2:
        k1, k2 = sorted(cvs_map.keys())[:2]
        per_map = (
            dataset.get("cv_periodic", {})
            if isinstance(dataset.get("cv_periodic", {}), dict)
            else {}
        )
        periodic = (bool(per_map.get(k1, False)), bool(per_map.get(k2, False)))
        return (
            np.asarray(cvs_map[k1], dtype=float),
            np.asarray(cvs_map[k2], dtype=float),
            (str(k1), str(k2)),
            periodic,
        )
    return None


def _extract_cvs_from_dict_X(
    dataset: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[str, str], Tuple[bool, bool]]]:
    if "X" in dataset:
        X = np.asarray(dataset["X"], dtype=float)
        if X.ndim == 2 and X.shape[1] >= 2:
            names = dataset.get("cv_names", ("cv0", "cv1"))
            if not isinstance(names, (list, tuple)) or len(names) < 2:
                names = ("cv0", "cv1")
            periodic = dataset.get("periodic", (False, False))
            if not isinstance(periodic, (list, tuple)) or len(periodic) < 2:
                periodic = (False, False)
            return (
                X[:, 0],
                X[:, 1],
                (str(names[0]), str(names[1])),
                (bool(periodic[0]), bool(periodic[1])),
            )
    return None


def _extract_cvs(
    dataset: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[str, str], Tuple[bool, bool]]]:
    """Extract a pair of continuous CVs and their names/periodicity if present.

    Returns (cv1, cv2, (name1, name2), (per1, per2)) or None.
    """
    # Dict-like inputs
    if isinstance(dataset, dict):
        out = _extract_cvs_from_dict_explicit(dataset)
        if out is not None:
            return out
        out = _extract_cvs_from_dict_map(dataset)
        if out is not None:
            return out
        out = _extract_cvs_from_dict_X(dataset)
        if out is not None:
            return out
    # Object-like inputs
    if hasattr(dataset, "cv1") and hasattr(dataset, "cv2"):
        return (
            np.asarray(getattr(dataset, "cv1"), dtype=float),
            np.asarray(getattr(dataset, "cv2"), dtype=float),
            ("cv1", "cv2"),
            (False, False),
        )
    cvs_attr = getattr(dataset, "cvs", None)
    if isinstance(cvs_attr, dict) and len(cvs_attr) >= 2:
        k1, k2 = sorted(cvs_attr.keys())[:2]
        return (
            np.asarray(cvs_attr[k1], dtype=float),
            np.asarray(cvs_attr[k2], dtype=float),
            (str(k1), str(k2)),
            (False, False),
        )
    X_attr = getattr(dataset, "X", None)
    if X_attr is not None:
        X = np.asarray(X_attr, dtype=float)
        if X.ndim == 2 and X.shape[1] >= 2:
            return (X[:, 0], X[:, 1], ("cv0", "cv1"), (False, False))
    return None


def _dh_update_arr(h, a: np.ndarray) -> None:
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())


def _hash_mapping_dispatch_into(h, k: str, v: Any) -> None:
    if k in {"dtrajs", "bias_matrices"} and isinstance(v, list):
        for arr in v:
            _dh_update_arr(h, np.asarray(arr))
        return
    if k in {"X", "cv1", "cv2"}:
        _dh_update_arr(h, np.asarray(v))
        return
    if k == "cvs" and isinstance(v, dict):
        for kk in sorted(v.keys()):
            _dh_update_arr(h, np.asarray(v[kk]))
        return
    if k in {"temperatures"} and isinstance(v, (list, tuple, np.ndarray)):
        _dh_update_arr(h, np.asarray(v))
        return
    if isinstance(v, (int, float, str, bool)):
        h.update(str(v).encode())


def _hash_mapping_into(h, d: Dict[str, Any]) -> None:
    for k in sorted(d.keys()):
        _hash_mapping_dispatch_into(h, k, d[k])


def _hash_object_into(h, obj: Any) -> None:
    if hasattr(obj, "dtrajs"):
        for arr in getattr(obj, "dtrajs"):
            _dh_update_arr(h, np.asarray(arr))
    if hasattr(obj, "X"):
        _dh_update_arr(h, np.asarray(getattr(obj, "X")))


def _dataset_hash(dataset: Any) -> str:
    """Compute a stable sha256 over core numerical inputs in the dataset."""
    h = sha256()
    if isinstance(dataset, dict):
        _hash_mapping_into(h, dataset)
    else:
        _hash_object_into(h, dataset)
    return h.hexdigest()


def default_fes_builder(
    dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Any | None:
    """Default FES builder using pmarlo.fes.surfaces.generate_2d_fes."""
    cv_pair = _extract_cvs(dataset)
    if cv_pair is None:
        return None
    from pmarlo.fes.surfaces import generate_2d_fes

    cv1, cv2, names, periodic = cv_pair
    bx, by = 32, 32
    if applied.bins and isinstance(applied.bins, dict):
        if names[0] in applied.bins and names[1] in applied.bins:
            bx = int(applied.bins[names[0]])
            by = int(applied.bins[names[1]])
        else:
            vals = [int(v) for v in applied.bins.values()]
            if len(vals) >= 2:
                bx, by = vals[0], vals[1]
    elif isinstance(applied.bins, tuple) and len(applied.bins) == 2:  # type: ignore[truthy-bool]
        bx, by = int(applied.bins[0]), int(applied.bins[1])  # type: ignore[index]
    elif opts.max_bins:
        bx = by = int(opts.max_bins)

    fes_notes = {}
    if isinstance(applied.notes, dict):
        fes_notes = dict(applied.notes.get("fes", {}))
    smooth = bool(fes_notes.get("smooth", False))
    inpaint = bool(fes_notes.get("inpaint", False))
    min_count = int(fes_notes.get("min_count", 1))
    kde_bw_deg = tuple(fes_notes.get("kde_bw_deg", (20.0, 20.0)))  # type: ignore[assignment]
    if not isinstance(kde_bw_deg, (list, tuple)) or len(kde_bw_deg) != 2:
        kde_bw_deg = (20.0, 20.0)
    temperature = float(opts.temperature) if opts.temperature else 300.0
    fes_obj = generate_2d_fes(
        cv1,
        cv2,
        bins=(bx, by),
        temperature=temperature,
        periodic=(bool(periodic[0]), bool(periodic[1])),
        smooth=smooth,
        inpaint=inpaint,
        min_count=min_count,
        kde_bw_deg=(float(kde_bw_deg[0]), float(kde_bw_deg[1])),
    )
    return {
        "result": fes_obj,
        "provenance": {
            "names": names,
            "bins": (bx, by),
            "periodic": periodic,
            "smooth": smooth,
            "inpaint": inpaint,
            "min_count": min_count,
            "kde_bw_deg": (float(kde_bw_deg[0]), float(kde_bw_deg[1])),
            "temperature": temperature,
        },
    }


def default_tram_builder(
    dataset: Any, opts: BuildOpts, applied: AppliedOpts
) -> Any | None:
    """Default TRAM builder using deeptime if available; returns compact payload."""
    if not (
        isinstance(dataset, dict) and "bias_matrices" in dataset and "dtrajs" in dataset
    ):
        return None
    dtrajs = [np.asarray(dt, dtype=int) for dt in dataset["dtrajs"]]
    bias = dataset["bias_matrices"]
    temps = dataset.get("temperatures", None)
    ref = int(dataset.get("tram_reference_index", 0))
    # guardrails
    if not isinstance(bias, list) or len(bias) != len(dtrajs):
        return {"method": "TRAM", "skipped": True, "reason": "bias/dtrajs_mismatch"}
    if temps is not None and len(temps) != len(dtrajs):
        return {"method": "TRAM", "skipped": True, "reason": "temps/dtrajs_mismatch"}
    try:
        from deeptime.markov.msm import TRAM, TRAMDataset  # type: ignore

        ds = TRAMDataset(dtrajs=dtrajs, bias_matrices=bias)
        tram = TRAM(lagtime=int(max(1, int(applied.lag or 1))), count_mode="sliding")
        tram_model = tram.fit(ds).fetch_model()
        msms = getattr(tram_model, "msms", None)
        cm_list = getattr(tram_model, "count_models", None)
        payload: Dict[str, Any] = {"method": "TRAM", "reference_index": ref}
        if isinstance(msms, list) and 0 <= ref < len(msms):
            msm_ref = msms[ref]
            payload.update(
                {
                    "transition_matrix": np.asarray(msm_ref.transition_matrix),
                    "stationary_distribution": (
                        np.asarray(msm_ref.stationary_distribution)
                        if getattr(msm_ref, "stationary_distribution", None) is not None
                        else None
                    ),
                    "count_matrix": (
                        np.asarray(cm_list[ref].count_matrix)
                        if isinstance(cm_list, list)
                        else None
                    ),
                }
            )
        else:
            payload.update({"skipped": True, "reason": "msm_not_exposed"})
        return payload
    except Exception as _exc:  # pragma: no cover - optional dependency
        return {"method": "TRAM", "skipped": True, "reason": f"deps_or_fit:{_exc}"}


def _normalize_numbers(obj: Any) -> Any:
    """Round floats for stable digests; leave ints/others unchanged."""
    if isinstance(obj, float):
        return round(obj, 12)
    if isinstance(obj, list):
        return [_normalize_numbers(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_normalize_numbers(list(obj)))
    if isinstance(obj, dict):
        return {k: _normalize_numbers(v) for k, v in obj.items()}
    return obj


# --- Helpers for LEARN_CV(DeepTICA) orchestration ---


def _deeptica_config_from_step(step: TransformStep, seed: Optional[int]):
    params = dict(step.params)
    from pmarlo.cv.deeptica import DeepTICAConfig  # type: ignore[import-not-found]

    return DeepTICAConfig(
        lag=int(params["lag"]),
        n_out=int(params.get("n_out", 2)),
        hidden=tuple(int(x) for x in params.get("hidden", (64, 64))),
        activation=str(params.get("activation", "tanh")),
        learning_rate=float(params.get("learning_rate", 1e-3)),
        batch_size=int(params.get("batch_size", 4096)),
        max_epochs=int(params.get("max_epochs", 200)),
        early_stopping=int(params.get("early_stopping", 20)),
        seed=int(seed or 0),
        reweight_mode=str(params.get("reweight_mode", "scaled_time")),
    )


def _collect_shards_info_from_dataset(transformed: Any) -> List[dict]:
    if isinstance(transformed, dict) and "__shards__" in transformed:
        return list(transformed["__shards__"])  # type: ignore[return-value]
    return []


def _build_records_for_pairs(
    transformed: Any, shards_info: List[dict], default_temp: float
) -> List[tuple[Any, Any, Any, Any]]:
    records: List[tuple[Any, Any, Any, Any]] = []
    if shards_info:
        X_all = np.asarray(transformed.get("X"), dtype=np.float64)
        for s in shards_info:
            if isinstance(s, dict) and {"start", "stop"}.issubset(s.keys()):
                start = int(s["start"])
                stop = int(s["stop"])
                X_block = X_all[start:stop]
                records.append(
                    (
                        X_block,
                        s.get("dtraj", None),
                        s.get("bias_potential", None),
                        s.get("temperature", None),
                    )
                )
            elif isinstance(s, (list, tuple)) and len(s) == 4:
                records.append((s[0], s[1], s[2], s[3]))
    elif isinstance(transformed, dict) and "X" in transformed:
        X_full = np.asarray(transformed["X"], dtype=np.float64)
        records = [(X_full, None, None, default_temp)]
    return records


def _deeptica_scaled_time_used(cfg, records: List[tuple[Any, Any, Any, Any]]) -> bool:
    return bool(
        str(cfg.reweight_mode).lower().startswith("scaled")
        and any(isinstance(r[2], np.ndarray) and r[2].size > 0 for r in records)
    )


def _deeptica_cache_key(pre_hash: str, cfg, scaled_time_used: bool) -> str:
    cfg_json = json.dumps(
        asdict(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(
        (pre_hash + cfg_json + f"|scaled:{int(scaled_time_used)}").encode("utf-8")
    ).hexdigest()


def _deeptica_model_paths(notes: Dict[str, Any], cache_key: str) -> tuple[Path, Path]:
    model_root = Path(notes.get("model_dir", "models")) / "deeptica" / cache_key
    model_root.mkdir(parents=True, exist_ok=True)
    return model_root, model_root / "deeptica"


def _load_or_train_model(X_list, pairs, cfg, model_base: Path):
    from pmarlo.cv.deeptica import (  # type: ignore[import-not-found]
        DeepTICAModel,
        train_deeptica,
    )

    use_cache = False
    have_ckpt = (
        model_base.with_suffix(".json").exists()
        and model_base.with_suffix(".pt").exists()
        and model_base.with_suffix(".scaler.pt").exists()
    )
    if have_ckpt:
        try:
            model = DeepTICAModel.load(model_base)
            use_cache = True
        except Exception:
            use_cache = False
    if not use_cache:
        model = train_deeptica(X_list, pairs, cfg, weights=None)
        try:
            model.save(model_base)
        except Exception:
            pass
    return model


def _ensure_ts_export(model, model_base: Path) -> None:
    try:
        if not model_base.with_suffix(".ts").exists():
            model.to_torchscript(model_base)
    except Exception:
        pass


def _apply_deeptica_model(
    model, X_list: List[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    X_concat = np.concatenate(X_list, axis=0).astype(np.float64, copy=False)
    Z = model.transform(X_concat)
    return X_concat, Z


def _new_dataset_with_learned_cvs(
    transformed: Any, Z: np.ndarray
) -> tuple[dict, tuple[str, ...]]:
    new = dict(transformed)
    new_names = tuple(f"cv{i+1}" for i in range(int(Z.shape[1])))
    new["X"] = Z
    new["cv_names"] = new_names
    new["periodic"] = tuple(False for _ in new_names)
    return new, new_names


def _bins_cfg_from_applied_or_opts(
    applied: AppliedOpts, opts: BuildOpts, new_names: tuple[str, ...], Z: np.ndarray
) -> Dict[str, int]:
    bins_cfg: Dict[str, int] = {}
    if isinstance(applied.bins, dict):
        bins_cfg = {str(k): int(v) for k, v in applied.bins.items()}
    elif isinstance(applied.bins, tuple):  # type: ignore[truthy-bool]
        vals = [int(v) for v in applied.bins]  # type: ignore[union-attr]
        for i, name in enumerate(new_names):
            if i < len(vals):
                bins_cfg[name] = int(vals[i])
    elif opts.max_bins:
        for name in new_names:
            bins_cfg[name] = int(opts.max_bins)
    else:
        for i in range(Z.shape[1]):
            bins_cfg[f"cv{i+1}"] = 32
    return bins_cfg


def _compute_edges_for_Z(
    Z: np.ndarray, new_names: tuple[str, ...], bins_cfg: Dict[str, int]
) -> Dict[str, np.ndarray]:
    edges: Dict[str, np.ndarray] = {}
    for i, name in enumerate(new_names):
        a = float(np.nanmin(Z[:, i]))
        b = float(np.nanmax(Z[:, i]))
        if np.isclose(a, b):
            b = a + 1e-8
        nb = int(bins_cfg.get(name, 32))
        edges[name] = np.linspace(a, b, nb + 1)
    return edges


def _write_plumed_and_ts_hash(
    model, model_root: Path, model_base: Path
) -> Optional[str]:
    try:
        snippet = model.plumed_snippet(model_base)
        (model_root / "plumed_deeptica.dat").write_text(snippet, encoding="utf-8")
    except Exception:
        pass
    try:
        p = model_base.with_suffix(".ts")
        if p.exists():
            with open(p, "rb") as fh:
                return sha256(fh.read()).hexdigest()
    except Exception:
        return None
    return None


def _relative_path_str(p: Path) -> str:
    try:
        return str(p.relative_to(Path.cwd()))
    except Exception:
        return str(p)


def _update_applied_with_mlcv(
    applied: AppliedOpts,
    cfg,
    _mlc: Any,
    _torch: Any,
    scaled_time_used: bool,
    pairs,
    X_concat: np.ndarray,
    ts_hash: Optional[str],
    cache_key: str,
    rel_model: str,
    edges: Dict[str, np.ndarray],
) -> AppliedOpts:
    notes = dict(applied.notes)
    notes["mlcv"] = {
        "method": "deeptica",
        "config": asdict(cfg),
        "seed": int(cfg.seed),
        "versions": {
            "mlcolvar": (
                getattr(_mlc, "__version__", None) if _mlc is not None else None
            ),
            "torch": getattr(_torch, "__version__", None),
            "numpy": np.__version__,
        },
        "scaled_time_used": bool(scaled_time_used),
        "pairs_count": int(pairs[0].size),
        "frames_count": int(X_concat.shape[0]),
        "torchscript_sha256": ts_hash,
        "cache_key": cache_key,
        "model_relpath": rel_model,
    }
    notes["cv_bin_edges"] = {k: v.tolist() for k, v in edges.items()}
    return replace(applied, notes=notes)


def _record_mlcv_artifacts(
    br_artifacts: Dict[str, Any],
    model_base: Path,
    rel_model: str,
    ts_hash: Optional[str],
    cache_key: str,
) -> None:
    try:
        files = []
        for suf in (".json", ".pt", ".scaler.pt", ".ts", ".dat"):
            f = model_base.with_suffix(suf)
            if f.exists():
                files.append(f.name)
        br_artifacts["mlcv_deeptica"] = {
            "model_dir": rel_model,
            "files": files,
            "torchscript_sha256": ts_hash,
            "cache_key": cache_key,
        }
    except Exception:
        pass


def _perform_learn_cv(
    transformed: Any,
    plan: TransformPlan,
    opts: BuildOpts,
    applied: AppliedOpts,
    br_artifacts: Dict[str, Any],
) -> tuple[Any, AppliedOpts, Dict[str, Any]]:
    """Handle LEARN_CV step (DeepTICA) if present; otherwise passthrough."""
    pre_hash = _dataset_hash(transformed)
    for step in getattr(plan, "steps", ()):  # type: ignore[attr-defined]
        if getattr(step, "name", "") != "LEARN_CV":
            continue
        if str(step.params.get("method", "")).lower() != "deeptica":
            continue
        try:
            import mlcolvar as _mlc  # type: ignore
        except Exception:  # pragma: no cover - optional extra
            _mlc = None
        import torch as _torch  # type: ignore

        cfg = _deeptica_config_from_step(step, opts.seed)
        shards_info = _collect_shards_info_from_dataset(transformed)
        default_temp = (
            float(opts.temperature) if opts.temperature is not None else 300.0
        )
        records = _build_records_for_pairs(transformed, shards_info, default_temp)
        if not records:
            return transformed, applied, br_artifacts

        from pmarlo.cv.pairs import (
            make_training_pairs_from_shards,  # type: ignore[import-not-found]
        )

        X_list, pairs = make_training_pairs_from_shards(records, tau_scaled=cfg.lag)
        if len(X_list) == 0 or pairs[0].size == 0:
            return transformed, applied, br_artifacts

        scaled_used = _deeptica_scaled_time_used(cfg, records)
        cache_key = _deeptica_cache_key(pre_hash, cfg, scaled_used)
        model_root, model_base = _deeptica_model_paths(dict(applied.notes), cache_key)
        model = _load_or_train_model(X_list, pairs, cfg, model_base)
        _ensure_ts_export(model, model_base)
        X_concat, Z = _apply_deeptica_model(model, X_list)
        new, new_names = _new_dataset_with_learned_cvs(transformed, Z)
        bins_cfg = _bins_cfg_from_applied_or_opts(applied, opts, new_names, Z)
        edges = _compute_edges_for_Z(Z, new_names, bins_cfg)
        ts_hash = _write_plumed_and_ts_hash(model, model_root, model_base)
        rel_model = _relative_path_str(model_root)
        applied = _update_applied_with_mlcv(
            applied,
            cfg,
            _mlc,
            _torch,
            scaled_used,
            pairs,
            X_concat,
            ts_hash,
            cache_key,
            rel_model,
            edges,
        )
        transformed = new
        _record_mlcv_artifacts(br_artifacts, model_base, rel_model, ts_hash, cache_key)
    return transformed, applied, br_artifacts


def build_result(
    dataset: Any,
    *,
    opts: BuildOpts,
    plan: TransformPlan,
    applied: AppliedOpts,
    fes_builder: Callable[
        [Any, BuildOpts, AppliedOpts], Any | None
    ] = default_fes_builder,
    tram_builder: Callable[
        [Any, BuildOpts, AppliedOpts], Any | None
    ] = default_tram_builder,
    progress_callback: Optional[ProgressCB] = None,
) -> BuildResult:
    """
    Standardized builder entry point.

    Parameters
    ----------
    dataset:
        Raw inputs (e.g., CVs and/or discrete trajectories).
    opts:
        Declarative build options (candidates, toggles, environment hints).
    plan:
        Transformation plan chosen for this dataset/environment.
    applied:
        Final resolved options actually applied (selected lag, bins, etc.).

    Returns
    -------
    BuildResult
        Envelope with provenance and the constructed analysis payload(s).
    """
    _set_seed(opts.seed)

    # 0) Optional ML-CV learning step(s) handled prior to other transforms.
    #    Currently supports LEARN_CV(method="deeptica").
    transformed = dataset
    br_artifacts: Dict[str, Any] = {}
    try:
        transformed, applied, br_artifacts = _perform_learn_cv(
            transformed, plan, opts, applied, br_artifacts
        )
    except ImportError:  # pragma: no cover - optional extra missing
        # If extras are missing, leave dataset unchanged; provenance will not include mlcv
        pass

    # 1) Apply remaining transform plan to data (no-ops by default, but explicit).
    #    Emit aggregate_* events if a progress callback is provided.
    if progress_callback is None:
        transformed = apply_transform_plan(transformed, plan)
    else:
        transformed = _apply_plan(plan, transformed, progress_callback)

    # 2) Build MSM if discrete trajectories are provided
    dtrajs = _extract_dtrajs(transformed)
    T: Optional[np.ndarray] = None
    pi: Optional[np.ndarray] = None
    if dtrajs:
        lag = int(applied.lag) if applied.lag is not None else 1
        n_states = opts.n_states
        T, pi = msm_bridge.build_simple_msm(
            dtrajs=dtrajs, n_states=n_states, lag=lag, count_mode=opts.count_mode
        )
    # 3) Build FES via injectable strategy
    fes_payload = fes_builder(transformed, opts, applied)
    fes_obj: Any | None = None
    fes_prov: Optional[Dict[str, Any]] = None
    if isinstance(fes_payload, dict) and "result" in fes_payload:
        fes_obj = fes_payload.get("result")
        prov = fes_payload.get("provenance", {})
        fes_prov = prov
        # record transform step names applied prior to FES
        fes_prov["transform_steps"] = [s.name for s in plan.steps]
        # merge structured provenance into applied.notes["fes"] for now
        new_notes = dict(applied.notes)
        new_notes["fes"] = prov
        applied = replace(applied, notes=new_notes)

    # 4) Optional TRAM/dTRAM if bias info present
    tram_obj: Any | None = tram_builder(transformed, opts, applied)

    # Compute provenance digest
    def _stable_json(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)

    md_plan = [
        {"name": s.name, "params": {k: s.params[k] for k in sorted(s.params)}}
        for s in plan.steps
    ]
    md_applied = _normalize_numbers(asdict(applied))
    md_opts = _normalize_numbers(asdict(opts))
    dataset_hash = _dataset_hash(transformed)
    digest = sha256(
        (
            _stable_json(md_plan)
            + _stable_json(md_applied)
            + _stable_json(md_opts)
            + dataset_hash
        ).encode("utf-8")
    ).hexdigest()

    metadata = RunMetadata(
        transform_plan=plan.steps,
        applied_opts=applied,
        build_opts=opts,
        dataset_hash=dataset_hash,
        digest=digest,
        fes=fes_prov,
    )
    result = BuildResult(
        metadata=metadata,
        transition_matrix=T,
        stationary_distribution=pi,
        fes=fes_obj,
        tram=tram_obj,
        artifacts=br_artifacts or None,
    )
    # Attach explicit flags for clarity
    has_tram = bool(isinstance(tram_obj, dict) and not tram_obj.get("skipped", False))
    result.flags = {
        "has_msm": T is not None,
        "has_fes": fes_obj is not None,
        "has_tram": has_tram,
    }
    if fes_obj is None:
        # diagnose why
        if _extract_cvs(transformed) is None:
            result.messages.append("fes_skipped:no_cvs")
    if isinstance(tram_obj, dict) and tram_obj.get("skipped", False):
        result.messages.append(f"tram_skipped:{tram_obj.get('reason', 'unknown')}")
    return result
