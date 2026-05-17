"""Collective variable (CV) training utilities."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["train_cv_model"]


def train_cv_model(
    features: np.ndarray,
    lag_time: int = 1,
    n_components: int = 2,
    method: str = "tica",
    **kwargs: Any,
) -> Any:
    """Train a collective-variable model on trajectory features.

    Parameters
    ----------
    features:
        Array of shape ``(n_frames, n_features)``.
    lag_time:
        Lag time in frames used for TICA / DeepTICA.
    n_components:
        Number of output CV dimensions.
    method:
        ``"tica"`` (default, uses deeptime) or ``"deeptica"`` (uses
        the pmarlo DeepTICA trainer).
    **kwargs:
        Forwarded to the underlying estimator.

    Returns
    -------
    Fitted model with a ``.transform(features)`` method.
    """
    if method == "tica":
        from deeptime.decomposition import TICA

        estimator = TICA(lagtime=lag_time, dim=n_components, **kwargs)
        model = estimator.fit(features).fetch_model()
        logger.info(
            "TICA trained: %d components, lag=%d frames", n_components, lag_time
        )
        return model

    if method == "deeptica":
        from pmarlo.features.deeptica import DeepTICAConfig, train_deeptica

        config = DeepTICAConfig(
            n_components=n_components,
            lag_time=lag_time,
            **kwargs,
        )
        model = train_deeptica(features, config)
        logger.info(
            "DeepTICA trained: %d components, lag=%d frames", n_components, lag_time
        )
        return model

    raise ValueError(f"Unknown CV method {method!r}. Choose 'tica' or 'deeptica'.")
