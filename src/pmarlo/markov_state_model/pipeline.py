from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

# Runtime import for actual usage
from .enhanced_msm import EnhancedMSM, EnhancedMSMProtocol

SupportsMSMPipeline = EnhancedMSMProtocol


def run_complete_msm_analysis(
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    output_dir: str | Path,
    n_states: int | Literal["auto"] = 100,
    lag_time: int = 20,
    feature_type: str = "phi_psi",
    temperatures: Optional[List[float]] = None,
    stride: int = 1,
    atom_selection: str | List[int] | None = None,
    chunk_size: int = 1000,
    ignore_trajectory_errors: bool = False,
) -> EnhancedMSMProtocol:
    msm = _create_msm(
        trajectory_files,
        topology_file,
        temperatures,
        output_dir,
        ignore_trajectory_errors,
    )
    msm_pipeline: SupportsMSMPipeline = msm
    _load_and_featurize(
        msm_pipeline,
        stride,
        atom_selection,
        chunk_size,
        feature_type,
        n_states,
        ignore_trajectory_errors,
    )
    _build_and_analyze(msm_pipeline, temperatures, lag_time)
    _emit_plots(msm_pipeline)
    return msm


def _create_msm(
    trajectory_files: Union[str, List[str]],
    topology_file: str,
    temperatures: Optional[List[float]],
    output_dir: str | Path,
    ignore_trajectory_errors: bool,
) -> SupportsMSMPipeline:
    return EnhancedMSM(
        trajectory_files=trajectory_files,
        topology_file=topology_file,
        temperatures=temperatures,
        output_dir=output_dir,
        ignore_trajectory_errors=ignore_trajectory_errors,
    )


def _load_and_featurize(
    msm: SupportsMSMPipeline,
    stride: int,
    atom_selection: str | List[int] | None,
    chunk_size: int,
    feature_type: str,
    n_states: int | Literal["auto"],
    ignore_trajectory_errors: bool,
) -> None:
    msm.load_trajectories(
        stride=stride, atom_selection=atom_selection, chunk_size=chunk_size
    )
    _validate_loaded_data(msm, ignore_trajectory_errors)
    msm.compute_features(feature_type=feature_type)
    msm.cluster_features(n_states=n_states)


def _build_and_analyze(
    msm: SupportsMSMPipeline,
    temperatures: Optional[List[float]],
    lag_time: int,
) -> None:
    method = (
        "tram" if temperatures is not None and len(temperatures) > 1 else "standard"
    )
    msm.build_msm(lag_time=lag_time, method=method)
    msm.compute_implied_timescales()
    msm.generate_free_energy_surface(cv1_name="CV1", cv2_name="CV2")
    msm.create_state_table()
    msm.extract_representative_structures()
    msm.save_analysis_results()


def _emit_plots(msm: SupportsMSMPipeline) -> None:
    msm.plot_free_energy_surface(save_file="free_energy_surface")
    msm.plot_implied_timescales(save_file="implied_timescales")
    msm.plot_implied_rates(save_file="implied_rates")
    msm.plot_free_energy_profile(save_file="free_energy_profile")
    msm.plot_ck_test(save_file="ck_plot", n_macrostates=3, factors=[2, 3, 4])


def _validate_loaded_data(
    msm: SupportsMSMPipeline, ignore_trajectory_errors: bool
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
