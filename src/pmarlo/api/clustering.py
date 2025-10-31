import logging
import numpy as np

from typing import Literal

from pmarlo.markov_state_model.clustering import (
        cluster_microstates as _cluster_microstates,
    )

logger = logging.getLogger("pmarlo")

def cluster_microstates(
    Y: np.ndarray,
    method: Literal["auto", "minibatchkmeans", "kmeans"] = "auto",
    n_states: int | Literal["auto"] = "auto",
    random_state: int | None = 42,
    minibatch_threshold: int = 5_000_000,
    **kwargs,
) -> np.ndarray:
    """Public wrapper around :func:`cluster.micro.cluster_microstates`.

    Parameters
    ----------
    Y:
        Reduced feature array.
    method:
        Clustering algorithm to use.  ``"auto"`` selects
        ``MiniBatchKMeans`` when the dataset size exceeds
        ``minibatch_threshold``.
    n_states:
        Number of states or ``"auto"`` to select via silhouette.
    random_state:
        Seed for deterministic clustering.  When ``None`` the global NumPy
        random state is used.
    minibatch_threshold:
        Product of frames and features above which ``MiniBatchKMeans`` is used
        when ``method="auto"``.

    Returns
    -------
    np.ndarray
        Integer labels per frame.
    """

    logger.info(
        "[clustering] Starting microstate clustering: shape=%s, method=%s, n_states=%s, random_state=%s",
        tuple(Y.shape),
        method,
        n_states,
        random_state,
    )

    result = _cluster_microstates(
        Y,
        method=method,
        n_states=n_states,
        random_state=random_state,
        minibatch_threshold=minibatch_threshold,
        **kwargs,
    )

    logger.info(
        "[clustering] Clustering complete: %d frames assigned to %d microstates",
        len(result.labels),
        result.n_states,
    )

    return result.labels