from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch


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

    _save_model_state(out, model, ckpt_path)
    _save_scaler_state(out, scaler)
    _write_metadata(out, metadata)
    _copy_training_history(out, model, metadata)

    return out


def _save_model_state(out: Path, model: Any, ckpt_path: str | None) -> None:
    state = getattr(model, "state_dict", None)
    state_dict = state() if callable(state) else {}
    torch.save({"state_dict": state_dict, "ckpt": ckpt_path}, out / "model.pt")


def _save_scaler_state(out: Path, scaler: Any) -> None:
    try:
        torch.save(scaler, out / "scaler.pt")
        return
    except Exception:
        pass
    try:
        import numpy as np  # local import to avoid hard dependency

        payload = {
            "mean": np.asarray(getattr(scaler, "mean_", None)),
            "std": np.asarray(getattr(scaler, "scale_", None)),
        }
        torch.save(payload, out / "scaler.pt")
    except Exception:
        pass


def _write_metadata(out: Path, metadata: dict[str, Any]) -> None:
    (out / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _copy_training_history(out: Path, model: Any, metadata: dict[str, Any]) -> None:
    try:
        metrics_path = _resolve_metrics_path(model, metadata)
        if metrics_path is None:
            return
        src = Path(metrics_path)
        if src.exists():
            shutil.copyfile(str(src), str(out / "history.csv"))
    except Exception:
        pass


def _resolve_metrics_path(model: Any, metadata: dict[str, Any]) -> Optional[str]:
    history = getattr(model, "training_history", {}) if hasattr(model, "training_history") else {}
    metrics_csv = history.get("metrics_csv") if isinstance(history, dict) else None
    if not metrics_csv:
        metrics_csv = metadata.get("metrics_csv")
    return str(metrics_csv) if metrics_csv else None
