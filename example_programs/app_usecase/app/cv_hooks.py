from __future__ import annotations

"""Optional MLCV hooks for Deep-TICA.

This module provides defaults and a small helper to prepare a LEARN_CV step
config. The heavy lifting is handled inside pmarlo.transform.build.
"""

from typing import Dict, Any


def default_deeptica_params(lag: int = 5) -> Dict[str, Any]:
    return {
        "lag": int(max(1, lag)),
        "n_out": 2,
        "hidden": (64, 64),
        "max_epochs": 200,
        "early_stopping": 20,
        "reweight_mode": "scaled_time",
    }
