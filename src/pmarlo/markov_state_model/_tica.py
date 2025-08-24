from __future__ import annotations

from typing import List, Optional

import numpy as np


class TICAMixin:
    def _maybe_apply_tica(self, n_components_hint: Optional[int], lag: int) -> None: ...

    def _split_features_by_trajectories(self) -> List[np.ndarray]: ...

    def _apply_deeptime_tica(
        self, *, Xs: List[np.ndarray], n_components: int, lag: int
    ) -> tuple[object, List[np.ndarray]]: ...

    def _discard_initial_frames_after_tica(self, lag: int) -> None: ...

    def _extract_tica_attributes(self, tica_model: object) -> None: ...

    def _fallback_discard(self, lag: int) -> None: ...
