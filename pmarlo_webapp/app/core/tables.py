from typing import Sequence, List, Dict, Any, Mapping

import numpy as np
import pandas as pd

from core.parsers import _format_tau_schedule

def _timescales_dataframe(
    lag_times: Sequence[int],
    timescales: Sequence[Sequence[float] | np.ndarray],
) -> pd.DataFrame:
    """Create a rectangular table of implied timescales across lags."""

    arrays = [np.asarray(ts, dtype=float).reshape(-1) for ts in timescales]
    max_len = max((arr.size for arr in arrays), default=0)
    columns = ["Lag (steps)"] + [f"Timescale {idx + 1}" for idx in range(max_len)]
    rows: List[Dict[str, Any]] = []
    for lag, arr in zip(lag_times, arrays):
        row: Dict[str, Any] = {"Lag (steps)": int(lag)}
        for idx in range(max_len):
            if idx < arr.size and np.isfinite(arr[idx]):
                row[f"Timescale {idx + 1}"] = float(arr[idx])
            else:
                row[f"Timescale {idx + 1}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)

def _shards_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        paths = entry.get("paths") or []
        created_at = entry.get("created_at")
        directory = entry.get("directory")
        rows.append(
            {
                "Index": idx,
                "Run ID": str(entry.get("run_id", "")),
                "Created": str(created_at) if created_at is not None else "",
                "Shards": len(paths) if isinstance(paths, Sequence) else None,
                "Frames": entry.get("n_frames"),
                "Stride": entry.get("stride"),
                "Hop": entry.get("hop_frames"),
                "Temperature (K)": entry.get("temperature"),
                "Directory": str(directory) if directory is not None else "",
            }
        )
    return pd.DataFrame(rows)

def _runs_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        traj_files = entry.get("traj_files") or []
        created_at = entry.get("created_at")
        run_dir = entry.get("run_dir")
        rows.append(
            {
                "Index": idx,
                "Run ID": str(entry.get("run_id", "")),
                "Created": str(created_at) if created_at is not None else "",
                "Steps": entry.get("steps"),
                "Trajectories": len(traj_files) if isinstance(traj_files, Sequence) else None,
                "Quick": bool(entry.get("quick", False)),
                "Directory": str(run_dir) if run_dir is not None else "",
            }
        )
    return pd.DataFrame(rows)

def _models_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        tau_schedule = entry.get("tau_schedule") or []
        hidden = entry.get("hidden") or []
        created_at = entry.get("created_at")
        dataset_hash = entry.get("dataset_hash")
        bundle = entry.get("bundle")
        rows.append(
            {
                "Index": idx,
                "Created": str(created_at) if created_at is not None else "",
                "Lag": entry.get("lag"),
                "Seed": entry.get("seed"),
                "Tau schedule": _format_tau_schedule(tau_schedule) if tau_schedule else "",
                "Hidden layers": ", ".join(str(h) for h in hidden) if hidden else "",
                "Dataset hash": str(dataset_hash) if dataset_hash is not None else "",
                "Bundle": str(bundle) if bundle is not None else "",
            }
        )
    return pd.DataFrame(rows)


def _builds_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        created_at = entry.get("created_at")
        reweight_mode = entry.get("reweight_mode")
        bundle = entry.get("bundle")
        debug_dir = entry.get("debug_dir")
        rows.append(
            {
                "Index": idx,
                "Created": str(created_at) if created_at is not None else "",
                "Lag": entry.get("lag"),
                "Microstates": entry.get("n_microstates"),
                "Reweight": str(reweight_mode) if reweight_mode is not None else "",
                "Bundle": str(bundle) if bundle is not None else "",
                "Debug dir": str(debug_dir) if debug_dir is not None else "",
            }
        )
    return pd.DataFrame(rows)


def _conformations_dataframe(entries: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        created_at = entry.get("created_at")
        output_dir = entry.get("output_dir")
        error = entry.get("error")
        rows.append(
            {
                "Index": idx,
                "Created": str(created_at) if created_at is not None else "",
                "Output": str(output_dir) if output_dir is not None else "",
                "Converged": entry.get("tpt_converged"),
                "Error": str(error) if error is not None else "",
            }
        )
    return pd.DataFrame(rows)

def _metrics_table(flags: Dict[str, object]) -> pd.DataFrame:
    """Render build flags in a tabular, Arrow-friendly representation.

    Streamlit converts the returned DataFrame into an Arrow table. Mixed dtypes
    within a column (for example booleans alongside strings) lead Arrow to infer
    an incompatible schema which subsequently raises an ``ArrowInvalid``. The
    build flags frequently contain boolean toggles together with nested
    structures such as diagnostic warning lists, so we normalise them into a
    flat table that stores display strings alongside their original type.
    """

    from collections.abc import Mapping
    from collections.abc import Sequence as _SequenceABC

    try:  # Local import to avoid an unconditional dependency at module import
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - NumPy is always available in practice
        np = None  # type: ignore

    def _coerce_scalar(val: object) -> object:
        if np is not None and isinstance(val, np.generic):
            return val.item()
        return val

    def _is_sequence(val: object) -> bool:
        return isinstance(val, _SequenceABC) and not isinstance(
            val, (str, bytes, bytearray)
        )

    def _iter_items(prefix: str, value: object):
        value = _coerce_scalar(value)

        if isinstance(value, Mapping):
            for sub_key, sub_val in value.items():
                next_prefix = f"{prefix}.{sub_key}" if prefix else str(sub_key)
                yield from _iter_items(next_prefix, sub_val)
            return

        if (
            np is not None
            and hasattr(value, "tolist")
            and not isinstance(value, (str, bytes, bytearray))
        ):
            try:
                value = value.tolist()
            except Exception:
                pass

        if _is_sequence(value):
            seq = list(value)
            if not seq:
                yield (prefix, "[]")
                return
            for idx, item in enumerate(seq, start=1):
                suffix = f"[{idx}]"
                next_prefix = f"{prefix}{suffix}" if prefix else suffix
                yield from _iter_items(next_prefix, item)
            return

        yield (prefix, value)

    def _format_value(val: object) -> tuple[str, str]:
        val = _coerce_scalar(val)
        if val is None:
            return "", "NoneType"
        if isinstance(val, bool):
            return ("True" if val else "False"), "bool"
        if isinstance(val, (int, float)):
            return f"{val}", type(val).__name__
        if isinstance(val, bytes):
            try:
                decoded = val.decode("utf-8")
            except Exception:
                decoded = val.decode("utf-8", errors="replace")
            return decoded, "bytes"
        return str(val), type(val).__name__

    rows: List[Dict[str, object]] = []
    for key, raw_value in flags.items():
        for metric_key, metric_value in _iter_items(key, raw_value):
            display, dtype_name = _format_value(metric_value)
            rows.append(
                {
                    "metric": metric_key,
                    "value": display,
                    "value_type": dtype_name,
                }
            )

    if not rows:
        return pd.DataFrame({"metric": [], "value": [], "value_type": []})

    df = pd.DataFrame(rows)
    df["value"] = pd.Series(df["value"], dtype="string")
    df["value_type"] = pd.Series(df["value_type"], dtype="string")
    return df
