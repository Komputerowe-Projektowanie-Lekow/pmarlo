from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class StatesMixin:
    def create_state_table(self) -> pd.DataFrame: ...

    def extract_representative_structures(self, save_pdb: bool = True): ...

    def _count_frames_per_state(self) -> tuple[np.ndarray, int]: ...

    def _bootstrap_free_energy_errors(
        self, counts: np.ndarray, n_boot: int = 200
    ) -> np.ndarray: ...

    def _find_representatives(
        self,
    ) -> tuple[List[tuple[int, int]], List[Optional[np.ndarray]]]: ...

    def _pcca_lumping(self, n_macrostates: int = 4) -> Optional[np.ndarray]: ...
