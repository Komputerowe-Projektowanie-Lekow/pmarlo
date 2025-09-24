from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch


def _save_model_state(model: Any, ckpt_path: str | None, out: Path) -> None:
    state_dict = getattr(model, "state_dict", None)
    payload = {
        "state_dict": state_dict() if callable(state_dict) else {},
        "ckpt": ckpt_path,
    }
    torch.save(payload, out / "model.pt")


def _save_scaler(scaler: Any, out: Path) -> None:
    try:
        torch.save(scaler, out / "scaler.pt")
        return
    except Exception:
        pass

    try:
        import numpy as np  # lazy import

        params = {
            "mean": np.asarray(getattr(scaler, "mean_", None)),
            "std": np.asarray(getattr(scaler, "scale_", None)),
        }
        torch.save(params, out / "scaler.pt")
    except Exception:
        pass


def _write_metadata(metadata: dict[str, Any], out: Path) -> None:
    (out / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _maybe_copy_history(model: Any, metadata: dict[str, Any], out: Path) -> None:
    metrics_csv = _resolve_metrics_path(model, metadata)
    if metrics_csv is None:
        return
    try:
        src = Path(str(metrics_csv))
        if src.exists():
            shutil.copyfile(str(src), out / "history.csv")
    except Exception:
        pass


def _resolve_metrics_path(model: Any, metadata: dict[str, Any]) -> Path | str | None:
    if hasattr(model, "training_history"):
        maybe_history = getattr(model, "training_history", {})
        metrics_csv = maybe_history.get("metrics_csv")
        if metrics_csv:
            return metrics_csv
    return metadata.get("metrics_csv")


def export_deeptica_bundle(
    model,
    scaler,
    metadata: dict[str, Any],
    ckpt_path: str | None,
    out_dir: str | os.PathLike[str],
) -> Path:
    """Export a neat DeepTICA model bundle.

    Files written:
      - model.pt (state_dict and checkpoint path)
      - scaler.pt (sklearn StandardScaler object or params)
      - config.json (metadata)
      - history.csv (if metrics.csv is available)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _save_model_state(model, ckpt_path, out)
    _save_scaler(scaler, out)
    _write_metadata(metadata, out)
    _maybe_copy_history(model, metadata, out)

    return out
