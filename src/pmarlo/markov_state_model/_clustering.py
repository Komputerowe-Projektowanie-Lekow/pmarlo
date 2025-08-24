from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from ..cluster.micro import ClusteringResult, cluster_microstates


class ClusteringMixin:
    def cluster_features(
        self,
        n_states: int | Literal["auto"] = "auto",
        algorithm: str = "kmeans",
        random_state: Optional[int] = None,
    ) -> None: ...
