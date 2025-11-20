"""Full EnhancedMSM workflow that relies on the optional ML stack."""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
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
from ._tpt import TPTMixin
from ._tram import TRAMMixin

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .enhanced_msm import EnhancedMSMProtocol


class _ComponentAdapter:
    """Routes attribute access from a component back to the owning EnhancedMSM."""

    def __init__(self, owner: "EnhancedMSM") -> None:
        object.__setattr__(self, "_owner", owner)

    @property
    def owner(self) -> "EnhancedMSM":
        return cast("EnhancedMSM", object.__getattribute__(self, "_owner"))

    def __getattr__(self, name: str):
        return getattr(self.owner, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_owner":
            object.__setattr__(self, name, value)
            return
        setattr(self.owner, name, value)


class _Loader(_ComponentAdapter, LoadingMixin):
    """Streaming trajectory loader."""


class _Featurizer(_ComponentAdapter, FeaturesMixin):
    """Feature computation pipeline."""


class _Clusterer(_ComponentAdapter, ClusteringMixin):
    """Discrete trajectory assignments."""


class _Estimator(_ComponentAdapter, EstimationMixin):
    """MSM estimation and bootstrapping."""


class _ITS(_ComponentAdapter, ITSMixin):
    """Implied timescale analysis."""


class _CK(_ComponentAdapter, CKMixin):
    """Chapman-Kolmogorov validation."""


class _FES(_ComponentAdapter, FESMixin):
    """Free energy surface utilities."""


class _TPT(_ComponentAdapter, TPTMixin):
    """Transition path theory routines."""


class _Plots(_ComponentAdapter, PlotsMixin):
    """Plot rendering helpers."""


class _States(_ComponentAdapter, StatesMixin):
    """State table and representative extraction."""


class _TRAM(_ComponentAdapter, TRAMMixin):
    """TRAM-specific workflows."""


class _Exporter(_ComponentAdapter, ExportMixin):
    """Disk export helpers."""


class EnhancedMSM(MSMBase):
    """Concrete EnhancedMSM composed of focused components instead of mixins."""

    def __init__(
        self,
        *,
        trajectory_files: Optional[Union[str, List[str]]] = None,
        topology_file: Optional[str] = None,
        temperatures: Optional[List[float]] = None,
        output_dir: str | Path | None = None,
        random_state: Optional[int] = 42,
        ignore_trajectory_errors: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(
            trajectory_files=trajectory_files,
            topology_file=topology_file,
            temperatures=temperatures,
            output_dir=output_dir,
            random_state=random_state,
            ignore_trajectory_errors=ignore_trajectory_errors,
        )
        # Dedicated components keep namespaces isolated while enabling focused tests.
        self.loader = _Loader(self)
        self.featurizer = _Featurizer(self)
        self.clusterer = _Clusterer(self)
        self.estimator = _Estimator(self)
        self.its = _ITS(self)
        self.ck = _CK(self)
        self.fes = _FES(self)
        self.tpt = _TPT(self)
        self.plots = _Plots(self)
        self.states = _States(self)
        self.tram = _TRAM(self)
        self.exporter = _Exporter(self)
        self._components: tuple[_ComponentAdapter, ...] = (
            self.loader,
            self.featurizer,
            self.clusterer,
            self.estimator,
            self.its,
            self.ck,
            self.fes,
            self.tpt,
            self.plots,
            self.states,
            self.tram,
            self.exporter,
        )
        if kwargs:
            raise TypeError(
                f"Unexpected arguments for EnhancedMSM: {', '.join(kwargs.keys())}"
            )

    def __getattr__(self, name: str):  # pragma: no cover - delegation glue
        components: Iterable[_ComponentAdapter] = self.__dict__.get("_components", ())
        for component in components:
            if hasattr(component, name):
                return getattr(component, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name}")

    def __dir__(self) -> list[str]:  # pragma: no cover - developer ergonomics
        base = set(super().__dir__())
        components: Iterable[_ComponentAdapter] = self.__dict__.get("_components", ())
        for component in components:
            base.update(dir(component))
        return sorted(base)


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
