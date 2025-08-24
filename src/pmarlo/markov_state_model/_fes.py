from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FESMixin:
    def generate_free_energy_surface(
        self,
        cv1_name: str = "phi",
        cv2_name: str = "psi",
        bins: int = 50,
        temperature: float = 300.0,
    ) -> Dict[str, Any]: ...

    def _validate_fes_prerequisites(self) -> None: ...

    def _map_stationary_to_frame_weights(self) -> np.ndarray: ...

    def _extract_collective_variables(
        self, cv1_name: str, cv2_name: str
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    def _align_data_lengths(
        self,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def _compute_weighted_histogram(
        self,
        cv1_data: np.ndarray,
        cv2_data: np.ndarray,
        frame_weights_array: np.ndarray,
        bins: int,
        ranges: Optional[List[Tuple[float, float]]] = None,
        smooth_sigma: Optional[float] = None,
        periodic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def _histogram_to_free_energy(
        self, H: np.ndarray, temperature: float
    ) -> np.ndarray: ...

    def _store_fes_result(
        self,
        F: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        cv1_name: str,
        cv2_name: str,
        temperature: float,
    ) -> None: ...

    def save_phi_psi_scatter_diagnostics(
        self,
        *,
        max_residues: int = 6,
        exclude_special: bool = True,
        sample_per_residue: int = 2000,
        filename: str = "diagnostics_phi_psi_scatter.png",
    ) -> Optional[Path]: ...
