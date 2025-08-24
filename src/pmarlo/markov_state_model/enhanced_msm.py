from __future__ import annotations

"""
Enhanced MSM composed from modular mixins.

This file intentionally keeps only the orchestration/class composition to keep
the implementation modular. All logic lives in the corresponding mixin modules.
"""

from typing import List, Literal, Optional, Sequence, Union

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
    pass


def run_complete_msm_analysis(
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    output_dir: str = "output/msm_analysis",
    n_states: int | Literal["auto"] = 100,
    lag_time: int = 20,
    feature_type: str = "phi_psi",
    temperatures: Optional[List[float]] = None,
    stride: int = 1,
    atom_selection: str | Sequence[int] | None = None,
    chunk_size: int = 1000,
):
    msm = _initialize_msm(
        trajectory_files=trajectory_files,
        topology_file=topology_file,
        temperatures=temperatures,
        output_dir=output_dir,
    )

    _load_and_prepare_data(
        msm=msm,
        stride=stride,
        atom_selection=atom_selection,
        chunk_size=chunk_size,
        feature_type=feature_type,
        n_states=n_states,
    )

    _build_and_analyze_msm(msm=msm, lag_time=lag_time, temperatures=temperatures)

    _compute_optional_fes(msm=msm)

    _finalize_and_export(msm=msm)

    _render_plots_safely(msm=msm)

    return msm


def _initialize_msm(
    *,
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    temperatures: Optional[List[float]],
    output_dir: str,
) -> EnhancedMSM:
    return EnhancedMSM(
        trajectory_files=trajectory_files,
        topology_file=topology_file,
        temperatures=temperatures,
        output_dir=output_dir,
    )


def _load_and_prepare_data(
    *,
    msm: EnhancedMSM,
    stride: int,
    atom_selection: str | Sequence[int] | None,
    chunk_size: int,
    feature_type: str,
    n_states: int | Literal["auto"],
) -> None:
    msm.load_trajectories(
        stride=stride,
        atom_selection=atom_selection,
        chunk_size=chunk_size,
    )
    msm.compute_features(feature_type=feature_type)
    msm.cluster_features(n_states=n_states)


def _build_and_analyze_msm(
    *, msm: EnhancedMSM, lag_time: int, temperatures: Optional[List[float]]
) -> None:
    method = _select_estimation_method(temperatures)
    msm.build_msm(lag_time=lag_time, method=method)
    msm.compute_implied_timescales()


def _select_estimation_method(temperatures: Optional[List[float]]) -> str:
    if temperatures and len(temperatures) > 1:
        return "tram"
    return "standard"


def _compute_optional_fes(*, msm: EnhancedMSM) -> None:
    try:
        # Default to a generic CV1/CV2 FES to avoid angle assumptions
        msm.generate_free_energy_surface(cv1_name="CV1", cv2_name="CV2")
    except Exception:
        # Optional; ignore any backend availability issues
        pass


def _finalize_and_export(*, msm: EnhancedMSM) -> None:
    msm.create_state_table()
    msm.extract_representative_structures()
    msm.save_analysis_results()


def _render_plots_safely(*, msm: EnhancedMSM) -> None:
    _try_plot(lambda: msm.plot_free_energy_surface(save_file="free_energy_surface"))
    _try_plot(lambda: msm.plot_implied_timescales(save_file="implied_timescales"))
    _try_plot(lambda: msm.plot_implied_rates(save_file="implied_rates"))
    _try_plot(lambda: msm.plot_free_energy_profile(save_file="free_energy_profile"))
    _try_plot(
        lambda: msm.plot_ck_test(
            save_file="ck_plot", n_macrostates=3, factors=[2, 3, 4]
        )
    )


def _try_plot(plot_callable) -> None:
    try:
        plot_callable()
    except Exception:
        # Optional; plotting may fail in headless or limited environments
        pass
