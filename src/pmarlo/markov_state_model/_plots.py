from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


class PlotsMixin:
    def plot_free_energy_surface(
        self, save_file: Optional[str] = None, interactive: bool = False
    ) -> None: ...

    def plot_free_energy_profile(self, save_file: Optional[str] = None) -> None: ...

    def plot_implied_timescales(self, save_file: Optional[str] = None) -> None: ...

    def plot_implied_rates(self, save_file: Optional[str] = None) -> None: ...

    def plot_ck_test(
        self,
        save_file: str = "ck_plot.png",
        n_macrostates: int = 3,
        factors: Optional[List[int]] = None,
    ) -> Optional[Path]: ...
