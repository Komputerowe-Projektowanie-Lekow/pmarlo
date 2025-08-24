from __future__ import annotations

from typing import List, Literal, Optional, Union

from .enhanced_msm import EnhancedMSM


def run_complete_msm_analysis(
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    output_dir: str = "output/msm_analysis",
    n_states: int | Literal["auto"] = 100,
    lag_time: int = 20,
    feature_type: str = "phi_psi",
    temperatures: Optional[List[float]] = None,
    stride: int = 1,
    atom_selection: str | List[int] | None = None,
    chunk_size: int = 1000,
) -> EnhancedMSM: ...
