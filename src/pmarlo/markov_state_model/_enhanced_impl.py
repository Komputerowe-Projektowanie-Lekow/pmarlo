"""Full EnhancedMSM workflow that relies on the optional ML stack."""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
)

from ._base import MSMBase
from ._ck import CKMixin
from ._clustering import ClusteringMixin
from ._estimation import EstimationMixin
from ._export import ExportMixin
from ._features import FeaturesMixin
from ._fes import FESMixin
from ._its import ITSMixin
from ._loading import LoadingMixin
from ._plots import PlotsMixin
from ._states import StatesMixin
from ._tram import TRAMMixin

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .enhanced_msm import EnhancedMSMProtocol


class EnhancedMSM(
    LoadingMixin,
    FeaturesMixin,
    ClusteringMixin,
    EstimationMixin,
    ITSMixin,
    CKMixin,
    FESMixin,
    PlotsMixin,
    StatesMixin,
    TRAMMixin,
    ExportMixin,
    MSMBase,
):
    """Concrete EnhancedMSM implementation backed by the full dependency stack."""


def run_complete_msm_analysis(
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    output_dir: str | Path,
    n_states: int | Literal["auto"] = 100,
    lag_time: int = 20,
    feature_type: str = "phi_psi",
    temperatures: Optional[List[float]] = None,
    stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
    ignore_trajectory_errors: bool = False,
) -> EnhancedMSM:
    msm = _initialize_msm(
        trajectory_files=trajectory_files,
        topology_file=topology_file,
        temperatures=temperatures,
        output_dir=output_dir,
        ignore_trajectory_errors=ignore_trajectory_errors,
    )
    msm_protocol = cast("EnhancedMSMProtocol", msm)

    _load_and_prepare_data(
        msm=msm_protocol,
        stride=stride,
        atom_selection=atom_selection,
        chunk_size=chunk_size,
        feature_type=feature_type,
        n_states=n_states,
        ignore_trajectory_errors=ignore_trajectory_errors,
    )

    _build_and_analyze_msm(
        msm=msm_protocol, lag_time=lag_time, temperatures=temperatures
    )

    _compute_optional_fes(msm=msm_protocol)
    _finalize_and_export(msm=msm_protocol)
    _render_plots(msm=msm_protocol)

    return msm


def _initialize_msm(
    *,
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    temperatures: Optional[List[float]],
    output_dir: str,
    ignore_trajectory_errors: bool,
) -> EnhancedMSM:
    return EnhancedMSM(
        trajectory_files=trajectory_files,
        topology_file=topology_file,
        temperatures=temperatures,
        output_dir=output_dir,
        ignore_trajectory_errors=ignore_trajectory_errors,
    )


def _load_and_prepare_data(
    *,
    msm: "EnhancedMSMProtocol",
    stride: int,
    atom_selection: str | Sequence[int] | None,
    chunk_size: int,
    feature_type: str,
    n_states: int | Literal["auto"],
    ignore_trajectory_errors: bool,
) -> None:
    msm.load_trajectories(
        stride=stride,
        atom_selection=atom_selection,
        chunk_size=chunk_size,
    )
    _validate_loaded_data(msm=msm, ignore_trajectory_errors=ignore_trajectory_errors)
    msm.compute_features(feature_type=feature_type)
    msm.cluster_features(n_states=n_states)


def _build_and_analyze_msm(
    *,
    msm: "EnhancedMSMProtocol",
    lag_time: int,
    temperatures: Optional[List[float]],
) -> None:
    method = _select_estimation_method(temperatures)
    msm.build_msm(lag_time=lag_time, method=method)
    msm.compute_implied_timescales()


def _select_estimation_method(temperatures: Optional[List[float]]) -> str:
    if temperatures is not None and len(temperatures) > 1:
        return "tram"
    return "standard"


def _compute_optional_fes(*, msm: "EnhancedMSMProtocol") -> None:
    msm.generate_free_energy_surface(cv1_name="CV1", cv2_name="CV2")


def _finalize_and_export(*, msm: "EnhancedMSMProtocol") -> None:
    msm.create_state_table()
    msm.extract_representative_structures()
    msm.save_analysis_results()


def _render_plots(*, msm: "EnhancedMSMProtocol") -> None:
    msm.plot_free_energy_surface(save_file="free_energy_surface")
    msm.plot_implied_timescales(save_file="implied_timescales")
    msm.plot_implied_rates(save_file="implied_rates")
    msm.plot_free_energy_profile(save_file="free_energy_profile")
    msm.plot_ck_test(save_file="ck_plot", n_macrostates=3, factors=[2, 3, 4])


def _validate_loaded_data(
    *, msm: "EnhancedMSMProtocol", ignore_trajectory_errors: bool
) -> None:
    trajectories = getattr(msm, "trajectories", None)
    if not trajectories:
        reason = "No trajectory data were loaded."
        if ignore_trajectory_errors:
            raise RuntimeError(
                f"{reason} Verify trajectory file paths and integrity before rerunning."
            )
        raise RuntimeError(reason)

    total_frames = getattr(msm, "total_frames", None)
    if total_frames is None or int(total_frames) <= 0:
        raise RuntimeError(
            "Loaded trajectories contain no frames; aborting MSM/FES analysis."
        )


__all__ = ["EnhancedMSM", "run_complete_msm_analysis"]
