"""Enhanced MSM workflow orchestrator with optional lightweight fallback."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
    cast,
)


class EnhancedMSMProtocol(Protocol):
    def __init__(
        self,
        *,
        trajectory_files: str | Sequence[str] | None = ...,
        topology_file: str | None = ...,
        temperatures: Sequence[float] | None = ...,
        output_dir: str | None = ...,
        **kwargs: object,
    ) -> None:
        ...

    def load_trajectories(
        self,
        *,
        stride: int,
        atom_selection: str | Sequence[int] | None,
        chunk_size: int,
    ) -> None: ...

    def compute_features(
        self,
        feature_type: str = ...,
        n_features: Optional[int] = ...,
        feature_stride: int = ...,
        tica_lag: int = ...,
        tica_components: Optional[int] = ...,
        **kwargs: object,
    ) -> None: ...

    def cluster_features(self, *, n_states: int | Literal["auto"]) -> None: ...

    def build_msm(self, *, lag_time: int, method: str = "standard") -> None: ...

    def compute_implied_timescales(self) -> None: ...

    def generate_free_energy_surface(
        self,
        cv1_name: str = ...,
        cv2_name: str = ...,
        bins: int = ...,
        temperature: float = ...,
        **kwargs: object,
    ) -> Dict[str, Any]: ...

    def create_state_table(self) -> None: ...

    def extract_representative_structures(self) -> None: ...

    def save_analysis_results(self) -> None: ...

    def plot_free_energy_surface(self, *, save_file: str) -> None: ...

    def plot_implied_timescales(self, *, save_file: str) -> None: ...

    def plot_implied_rates(self, *, save_file: str) -> None: ...

    def plot_free_energy_profile(self, *, save_file: str) -> None: ...

    def plot_ck_test(
        self,
        save_file: str = ...,
        n_macrostates: int = ...,
        factors: Optional[List[int]] = ...,
        **kwargs: object,
    ) -> Optional[Path]: ...


_SKLEARN_SPEC = importlib.util.find_spec("sklearn")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._enhanced_impl import EnhancedMSM as _EnhancedMSMImpl
    from ._enhanced_impl import run_complete_msm_analysis as _run_complete_msm_analysis
else:
    if _SKLEARN_SPEC is None:  # pragma: no cover - exercised in minimal test envs
        from ._enhanced_stub import EnhancedMSMStub as _EnhancedMSMImpl
        from ._enhanced_stub import run_complete_msm_analysis as _run_complete_msm_analysis
    else:  # pragma: no cover - relies on optional ML stack
        from ._enhanced_impl import EnhancedMSM as _EnhancedMSMImpl
        from ._enhanced_impl import run_complete_msm_analysis as _run_complete_msm_analysis


class _RunAnalysisProto(Protocol):
    def __call__(
        self,
        trajectory_files: str | List[str],
        topology_file: str,
        output_dir: str = ...,
        n_states: int | Literal["auto"] = ...,
        lag_time: int = ...,
        feature_type: str = ...,
        temperatures: Optional[List[float]] = ...,
        stride: int = ...,
        atom_selection: str | Sequence[int] | None = ...,
        chunk_size: int = ...,
    ) -> EnhancedMSMProtocol:
        ...


EnhancedMSM: type[EnhancedMSMProtocol] = cast(type[EnhancedMSMProtocol], _EnhancedMSMImpl)
run_complete_msm_analysis: _RunAnalysisProto = cast(
    _RunAnalysisProto, _run_complete_msm_analysis
)


__all__ = ["EnhancedMSMProtocol", "EnhancedMSM", "run_complete_msm_analysis"]
