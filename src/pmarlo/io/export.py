from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

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

    # Model state
    torch.save(
        {
            "state_dict": (
                getattr(model, "state_dict")() if hasattr(model, "state_dict") else {}
            ),
            "ckpt": ckpt_path,
        },
        out / "model.pt",
    )

    # Scaler
    try:
        torch.save(scaler, out / "scaler.pt")
    except Exception:
        # Fallback to saving parameters
        try:
            import numpy as np  # lazy

            torch.save(
                {
                    "mean": np.asarray(getattr(scaler, "mean_", None)),
                    "std": np.asarray(getattr(scaler, "scale_", None)),
                },
                out / "scaler.pt",
            )
        except Exception:
            pass

    # Config/metadata
    (out / "config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Try copying metrics.csv as history.csv
    try:
        metrics_csv = None
        # Accept both attributes on model or metadata
        if hasattr(model, "training_history"):
            metrics_csv = getattr(model, "training_history", {}).get("metrics_csv")
        if not metrics_csv:
            metrics_csv = metadata.get("metrics_csv")
        if metrics_csv:
            src = Path(str(metrics_csv))
            if src.exists():
                shutil.copyfile(str(src), str(out / "history.csv"))
    except Exception:
        pass

    return out
