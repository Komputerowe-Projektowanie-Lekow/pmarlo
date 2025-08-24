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
) -> EnhancedMSM:
    msm = _create_msm(trajectory_files, topology_file, temperatures, output_dir)
    _load_and_featurize(msm, stride, atom_selection, chunk_size, feature_type, n_states)
    _build_and_analyze(msm, temperatures, lag_time)
    _emit_plots(msm)
    return msm


def _create_msm(
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


def _load_and_featurize(
    msm: EnhancedMSM,
    stride: int,
    atom_selection: str | List[int] | None,
    chunk_size: int,
    feature_type: str,
    n_states: int | Literal["auto"],
) -> None:
    msm.load_trajectories(
        stride=stride, atom_selection=atom_selection, chunk_size=chunk_size
    )
    msm.compute_features(feature_type=feature_type)
    msm.cluster_features(n_states=n_states)


def _build_and_analyze(
    msm: EnhancedMSM,
    temperatures: Optional[List[float]],
    lag_time: int,
) -> None:
    method = "tram" if temperatures and len(temperatures) > 1 else "standard"
    msm.build_msm(lag_time=lag_time, method=method)
    msm.compute_implied_timescales()
    try:
        msm.generate_free_energy_surface(cv1_name="CV1", cv2_name="CV2")
    except Exception:
        pass
    msm.create_state_table()
    msm.extract_representative_structures()
    msm.save_analysis_results()


def _emit_plots(msm: EnhancedMSM) -> None:
    for fn in (
        lambda: msm.plot_free_energy_surface(save_file="free_energy_surface"),
        lambda: msm.plot_implied_timescales(save_file="implied_timescales"),
        lambda: msm.plot_implied_rates(save_file="implied_rates"),
        lambda: msm.plot_free_energy_profile(save_file="free_energy_profile"),
        lambda: msm.plot_ck_test(
            save_file="ck_plot", n_macrostates=3, factors=[2, 3, 4]
        ),
    ):
        try:
            fn()
        except Exception:
            pass
