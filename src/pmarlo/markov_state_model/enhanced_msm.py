"""Enhanced MSM workflow orchestrator with optional lightweight fallback."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Union, cast


class EnhancedMSMProtocol(Protocol):
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

if _SKLEARN_SPEC is None:  # pragma: no cover - exercised in minimal test envs
    import numpy as np

    logger = logging.getLogger("pmarlo")

    class _EnhancedMSMStub(EnhancedMSMProtocol):
        """Minimal stub used when scikit-learn is unavailable.

        The stub provides enough surface area for unit tests that exercise frame
        accounting logic without pulling in the heavy clustering and estimation
        stack.  It accepts trajectories assigned directly to the ``trajectories``
        attribute and tracks the number of effective frames produced by
        :meth:`compute_features`.
        """

        def __init__(
            self,
            *,
            trajectory_files: Union[str, List[str]] | None = None,
            topology_file: str | None = None,
            temperatures: Optional[List[float]] = None,
            output_dir: str | None = None,
            **_: object,
        ) -> None:
            self.output_dir = output_dir
            self.trajectories: list[object] = []
            self._effective_frames = 0
            self._feature_stride = 1
            self._tica_lag = 0

        @property
        def effective_frames(self) -> int:
            return int(self._effective_frames)

        def compute_features(
            self,
            feature_type: str = "",
            n_features: Optional[int] = None,
            feature_stride: int = 1,
            tica_lag: int = 0,
            tica_components: Optional[int] = None,
            **_: object,
        ) -> None:
            stride = int(feature_stride or 1)
            if stride <= 0:
                stride = 1
            total_frames = 0
            for traj in self.trajectories:
                n_frames = getattr(traj, "n_frames", None)
                if n_frames is None and hasattr(traj, "xyz"):
                    n_frames = np.asarray(traj.xyz).shape[0]
                total_frames += int(n_frames or 0)
            processed = total_frames // stride
            self._feature_stride = stride
            self._tica_lag = int(max(0, tica_lag))
            self._effective_frames = max(0, processed - self._tica_lag)

        def build_msm(self, *, lag_time: int, **_: object) -> None:
            lag = int(max(0, lag_time))
            if self.effective_frames < lag:
                msg = f"effective frames after lag {lag}: {self.effective_frames}"
                logger.info(msg)
                raise ValueError(msg)

        # The full implementation exposes many additional methods.  The stub keeps
        # compatibility by defining no-op placeholders so callers that expect these
        # attributes do not fail loudly in the reduced environment.
        def compute_features_from_traj(
            self,
            *,
            feature_stride: int | None = None,
            tica_lag: int = 0,
            tica_components: int | None = None,
        ) -> None:
            stride_value = 1 if feature_stride is None else int(feature_stride)
            self.compute_features(
                feature_stride=stride_value,
                tica_lag=tica_lag,
                tica_components=tica_components,
            )

        # --- No-op placeholders for interface compatibility in minimal environments ---
        def load_trajectories(
            self,
            *,
            stride: int = 1,
            atom_selection: str | Sequence[int] | None = None,
            chunk_size: int = 1000,
        ) -> None:
            return None

        def cluster_features(
            self,
            *,
            n_states: int | Literal["auto"] = 100,
        ) -> None:
            return None

        def compute_implied_timescales(self) -> None:
            return None

        def generate_free_energy_surface(
            self,
            cv1_name: str = "CV1",
            cv2_name: str = "CV2",
            bins: int = 100,
            temperature: float = 300.0,
            **_: object,
        ) -> Dict[str, Any]:
            return {
                "cv1": cv1_name,
                "cv2": cv2_name,
                "bins": int(bins),
                "temperature": float(temperature),
                "surface": [],
            }

        def create_state_table(self) -> None:
            return None

        def extract_representative_structures(self) -> None:
            return None

        def save_analysis_results(self) -> None:
            return None

        def plot_free_energy_surface(self, *, save_file: str) -> None:
            return None

        def plot_implied_timescales(self, *, save_file: str) -> None:
            return None

        def plot_implied_rates(self, *, save_file: str) -> None:
            return None

        def plot_free_energy_profile(self, *, save_file: str) -> None:
            return None

        def plot_ck_test(
            self,
            save_file: str = "ck_plot",
            n_macrostates: int = 3,
            factors: Optional[List[int]] = None,
            **_: object,
        ) -> Optional[Path]:
            return None

    _EnhancedMSMStubClass = cast(type[EnhancedMSMProtocol], _EnhancedMSMStub)

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
    ) -> EnhancedMSM:
        raise ImportError(
            "EnhancedMSM full pipeline requires the optional scikit-learn dependency"
        )

else:  # pragma: no cover - relies on optional ML stack
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
    ) -> EnhancedMSM:
        msm = _initialize_msm(
            trajectory_files=trajectory_files,
            topology_file=topology_file,
            temperatures=temperatures,
            output_dir=output_dir,
        )
        msm_protocol = cast(EnhancedMSMProtocol, msm)

        _load_and_prepare_data(
            msm=msm_protocol,
            stride=stride,
            atom_selection=atom_selection,
            chunk_size=chunk_size,
            feature_type=feature_type,
            n_states=n_states,
        )

        _build_and_analyze_msm(
            msm=msm_protocol, lag_time=lag_time, temperatures=temperatures
        )

        _compute_optional_fes(msm=msm_protocol)

        _finalize_and_export(msm=msm_protocol)

        _render_plots_safely(msm=msm_protocol)

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
        msm: EnhancedMSMProtocol,
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
        *, msm: EnhancedMSMProtocol, lag_time: int, temperatures: Optional[List[float]]
    ) -> None:
        method = _select_estimation_method(temperatures)
        msm.build_msm(lag_time=lag_time, method=method)
        msm.compute_implied_timescales()

    def _select_estimation_method(temperatures: Optional[List[float]]) -> str:
        if temperatures and len(temperatures) > 1:
            return "tram"
        return "standard"

    def _compute_optional_fes(*, msm: EnhancedMSMProtocol) -> None:
        try:
            msm.generate_free_energy_surface(cv1_name="CV1", cv2_name="CV2")
        except Exception:
            pass

    def _finalize_and_export(*, msm: EnhancedMSMProtocol) -> None:
        msm.create_state_table()
        msm.extract_representative_structures()
        msm.save_analysis_results()

    def _render_plots_safely(*, msm: EnhancedMSMProtocol) -> None:
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
            pass


# Export unified interface
if _SKLEARN_SPEC is None:
    EnhancedMSM = _EnhancedMSMStubClass  # type: ignore[misc,assignment]
