from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Iterator, Sequence

if "numpy" not in sys.modules:
    numpy_stub = types.ModuleType("numpy")

    class _StubFloating:
        def __class_getitem__(cls, item: object) -> type:
            return float

    class _StubComplexFloating(_StubFloating):
        def __class_getitem__(cls, item: object) -> type:
            return complex

    numpy_stub.array = lambda data, dtype=None: list(data)
    numpy_stub.asarray = lambda data, dtype=None: list(data)
    numpy_stub.random = types.SimpleNamespace(seed=lambda _: None)
    numpy_stub.complexfloating = _StubComplexFloating
    numpy_stub.floating = _StubFloating
    numpy_typing = types.ModuleType("numpy.typing")
    numpy_typing.NDArray = object
    sys.modules["numpy"] = numpy_stub
    sys.modules["numpy.typing"] = numpy_typing

import numpy as np

if "mdtraj" not in sys.modules:
    mdtraj_stub = types.ModuleType("mdtraj")

    class _StubTrajectory:
        n_frames: int = 0

        def join(self, other: "_StubTrajectory") -> "_StubTrajectory":
            return self

    mdtraj_stub.Trajectory = _StubTrajectory
    sys.modules["mdtraj"] = mdtraj_stub

if "pandas" not in sys.modules:
    pandas_stub = types.ModuleType("pandas")

    class _StubDataFrame:  # pragma: no cover - minimal stub
        def __init__(self, *args: object, **kwargs: object) -> None:  # noqa: D401
            pass

    pandas_stub.DataFrame = _StubDataFrame
    sys.modules["pandas"] = pandas_stub

if "scipy" not in sys.modules:
    scipy_stub = types.ModuleType("scipy")
    sparse_mod = types.ModuleType("scipy.sparse")
    csgraph_mod = types.ModuleType("scipy.sparse.csgraph")

    def _connected_components(*_: object, **__: object) -> tuple[int, list[int]]:
        return 0, []

    csgraph_mod.connected_components = _connected_components
    sparse_mod.csc_matrix = lambda *args, **kwargs: args[0] if args else None
    sparse_mod.csr_matrix = lambda *args, **kwargs: args[0] if args else None
    sparse_mod.issparse = lambda _: False
    sparse_mod.save_npz = lambda *args, **kwargs: None
    sparse_mod.csgraph = csgraph_mod
    scipy_stub.sparse = sparse_mod
    sys.modules["scipy"] = scipy_stub
    sys.modules["scipy.sparse"] = sparse_mod
    sys.modules["scipy.sparse.csgraph"] = csgraph_mod

if "sklearn" not in sys.modules:
    sklearn_stub = types.ModuleType("sklearn")
    cluster_mod = types.ModuleType("sklearn.cluster")
    metrics_mod = types.ModuleType("sklearn.metrics")
    sklearn_stub.__path__ = []  # type: ignore[attr-defined]
    cluster_mod.__path__ = []  # type: ignore[attr-defined]
    metrics_mod.__path__ = []  # type: ignore[attr-defined]

    class _DummyKMeans:  # pragma: no cover - minimal stub
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, X: Sequence[object]) -> "_DummyKMeans":
            return self

        def predict(self, X: Sequence[object]) -> list[int]:
            return [0 for _ in X]

    cluster_mod.KMeans = _DummyKMeans
    cluster_mod.MiniBatchKMeans = _DummyKMeans
    metrics_mod.silhouette_score = lambda *args, **kwargs: 0.0
    sklearn_stub.cluster = cluster_mod
    sklearn_stub.metrics = metrics_mod
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.cluster"] = cluster_mod
    sys.modules["sklearn.metrics"] = metrics_mod

if "deeptime" not in sys.modules:
    deeptime_stub = types.ModuleType("deeptime")
    deeptime_stub.__path__ = []  # type: ignore[attr-defined]

    markov_mod = types.ModuleType("deeptime.markov")
    markov_mod.__path__ = []  # type: ignore[attr-defined]
    msm_mod = types.ModuleType("deeptime.markov.msm")
    msm_mod.__path__ = []  # type: ignore[attr-defined]
    tools_mod = types.ModuleType("deeptime.markov.tools")
    tools_mod.__path__ = []  # type: ignore[attr-defined]
    analysis_mod = types.ModuleType("deeptime.markov.tools.analysis")
    analysis_mod.__path__ = []  # type: ignore[attr-defined]
    estimation_mod = types.ModuleType("deeptime.markov.tools.estimation")
    estimation_mod.__path__ = []  # type: ignore[attr-defined]
    dense_mod = types.ModuleType("deeptime.markov.tools.estimation.dense")
    dense_mod.__path__ = []  # type: ignore[attr-defined]
    tm_mod = types.ModuleType(
        "deeptime.markov.tools.estimation.dense.transition_matrix"
    )
    tm_mod.__path__ = []  # type: ignore[attr-defined]
    decomposition_mod = types.ModuleType("deeptime.decomposition")
    decomposition_mod.__path__ = []  # type: ignore[attr-defined]
    plots_mod = types.ModuleType("deeptime.plots")
    plots_mod.__path__ = []  # type: ignore[attr-defined]
    clustering_mod = types.ModuleType("deeptime.clustering")
    clustering_mod.__path__ = []  # type: ignore[attr-defined]
    util_mod = types.ModuleType("deeptime.util")
    util_mod.__path__ = []  # type: ignore[attr-defined]
    validation_mod = types.ModuleType("deeptime.util.validation")
    validation_mod.__path__ = []  # type: ignore[attr-defined]

    class _StubEstimator:  # pragma: no cover - minimal stub
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, *args: object, **kwargs: object) -> "_StubEstimator":
            return self

    class _StubMSM(_StubEstimator):
        def count_model(self) -> "_StubMSM":  # pragma: no cover - minimal
            return self

        def transition_matrix(self) -> None:  # pragma: no cover - minimal
            return None

    class _StubPCCA(_StubEstimator):
        def assign(self, *args: object, **kwargs: object) -> list[int]:
            return []

    markov_mod.TransitionCountEstimator = _StubEstimator
    markov_mod.pcca = types.SimpleNamespace(PCCA=_StubPCCA)
    msm_mod.MaximumLikelihoodMSM = _StubMSM
    msm_mod.BayesianMSM = _StubMSM
    msm_mod.TRAM = _StubMSM
    msm_mod.TRAMDataset = _StubEstimator
    analysis_mod.eigenvalues = lambda *args, **kwargs: []
    analysis_mod.timescales = lambda *args, **kwargs: []
    analysis_mod.stationary_distribution = lambda *args, **kwargs: []
    analysis_mod.is_transition_matrix = lambda *args, **kwargs: True
    estimation_mod.dense = dense_mod
    dense_mod.transition_matrix = tm_mod
    tm_mod.TransitionMatrixEstimator = _StubEstimator
    tm_mod.transition_matrix_non_reversible = lambda *args, **kwargs: (
        args[0] if args else None
    )
    decomposition_mod.TICA = _StubEstimator
    decomposition_mod.VAMP = _StubEstimator

    class _StubKMeans(_StubEstimator):  # pragma: no cover - minimal stub
        def predict(self, X: Sequence[object]) -> list[int]:
            return [0 for _ in X]

        @property
        def cluster_centers_(self) -> None:
            return None

    clustering_mod.KMeans = _StubKMeans
    clustering_mod.MiniBatchKMeans = _StubKMeans

    plots_mod.plot_ck_test = lambda *args, **kwargs: None
    validation_mod.ck_test = lambda *args, **kwargs: None
    util_mod.validation = validation_mod

    deeptime_stub.markov = markov_mod
    deeptime_stub.decomposition = decomposition_mod
    deeptime_stub.plots = plots_mod
    deeptime_stub.util = util_mod
    sys.modules["deeptime"] = deeptime_stub
    sys.modules["deeptime.markov"] = markov_mod
    sys.modules["deeptime.markov.msm"] = msm_mod
    sys.modules["deeptime.markov.tools"] = tools_mod
    sys.modules["deeptime.markov.tools.analysis"] = analysis_mod
    sys.modules["deeptime.markov.tools.estimation"] = estimation_mod
    sys.modules["deeptime.markov.tools.estimation.dense"] = dense_mod
    sys.modules["deeptime.markov.tools.estimation.dense.transition_matrix"] = tm_mod
    sys.modules["deeptime.decomposition"] = decomposition_mod
    sys.modules["deeptime.plots"] = plots_mod
    sys.modules["deeptime.clustering"] = clustering_mod
    sys.modules["deeptime.util"] = util_mod
    sys.modules["deeptime.util.validation"] = validation_mod

import pytest

from pmarlo.markov_state_model._enhanced_impl import _select_estimation_method
from pmarlo.markov_state_model._loading import LoadingMixin
from pmarlo.markov_state_model.enhanced_msm import run_complete_msm_analysis
from pmarlo.markov_state_model.pipeline import _build_and_analyze


class _LoaderHarness(LoadingMixin):  # type: ignore[misc, valid-type]
    def __init__(self, *, traj: Path, topo: Path, ignore: bool) -> None:
        self.trajectory_files = [str(traj)]
        self.topology_file = str(topo)
        self.demux_metadata = None
        self.frame_stride = None
        self.time_per_frame_ps = None
        self.total_frames = None
        self.ignore_trajectory_errors = ignore
        self.logger = None

    def _update_total_frames(self) -> None:  # pragma: no cover - deterministic helper
        self.total_frames = sum(
            getattr(traj, "n_frames", 0) for traj in getattr(self, "trajectories", [])
        )


def test_loading_mixin_can_ignore_corrupt_trajectories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    traj_path = tmp_path / "traj.xtc"
    topo_path = tmp_path / "structure.pdb"
    traj_path.write_bytes(b"")
    topo_path.write_bytes(b"HEADER")

    def _iterload(*_: object, **__: object) -> Iterator[object]:
        raise RuntimeError("corrupt trajectory")

    monkeypatch.setattr("pmarlo.io.trajectory.iterload", _iterload)

    loader = _LoaderHarness(traj=traj_path, topo=topo_path, ignore=True)
    loader.load_trajectories()
    assert loader.trajectories == []
    assert loader.total_frames == 0


def test_loading_mixin_raises_when_not_ignoring_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    traj_path = tmp_path / "traj.xtc"
    topo_path = tmp_path / "structure.pdb"
    traj_path.write_bytes(b"")
    topo_path.write_bytes(b"HEADER")

    def _iterload(*_: object, **__: object) -> Iterator[object]:
        raise RuntimeError("corrupt trajectory")

    monkeypatch.setattr("pmarlo.io.trajectory.iterload", _iterload)

    loader = _LoaderHarness(traj=traj_path, topo=topo_path, ignore=False)
    with pytest.raises(RuntimeError, match="corrupt trajectory"):
        loader.load_trajectories()


class _NoOpEnhancedMSM:
    """Minimal EnhancedMSM stub that records load attempts and rejects further work."""

    def __init__(
        self,
        *,
        trajectory_files: Sequence[str] | str | None = None,
        topology_file: str | None = None,
        output_dir: str | Path | None = None,
        temperatures: Sequence[float] | None = None,
        ignore_trajectory_errors: bool = False,
        **_: object,
    ) -> None:
        if isinstance(trajectory_files, str):
            self.trajectory_files = [trajectory_files]
        else:
            self.trajectory_files = list(trajectory_files or [])
        self.topology_file = topology_file
        self.temperatures = list(temperatures or [])
        self.output_dir = Path(output_dir or Path.cwd())
        self.ignore_trajectory_errors = bool(ignore_trajectory_errors)
        self.trajectories: list[object] = []
        self.total_frames: int | None = 0

    def load_trajectories(self, **_: object) -> None:
        self.trajectories = []
        self.total_frames = 0

    def _fail(self, stage: str) -> None:
        raise AssertionError(f"{stage} should not be reached")

    def compute_features(self, **_: object) -> None:  # pragma: no cover - guard
        self._fail("features")

    def cluster_features(self, **_: object) -> None:  # pragma: no cover
        self._fail("clustering")

    def build_msm(self, **_: object) -> None:  # pragma: no cover
        self._fail("build")

    def compute_implied_timescales(self) -> None:  # pragma: no cover
        self._fail("ITS")

    def generate_free_energy_surface(self, **_: object) -> None:  # pragma: no cover
        self._fail("FES")

    def create_state_table(self) -> None:  # pragma: no cover
        self._fail("state table")

    def extract_representative_structures(self) -> None:  # pragma: no cover
        self._fail("representatives")

    def save_analysis_results(self) -> None:  # pragma: no cover
        self._fail("save")

    def plot_free_energy_surface(self, **_: object) -> None:  # pragma: no cover
        self._fail("plot")

    def plot_implied_timescales(self, **_: object) -> None:  # pragma: no cover
        self._fail("plot")

    def plot_implied_rates(self, **_: object) -> None:  # pragma: no cover
        self._fail("plot")

    def plot_free_energy_profile(self, **_: object) -> None:  # pragma: no cover
        self._fail("plot")

    def plot_ck_test(self, **_: object) -> None:  # pragma: no cover
        self._fail("plot")


def test_run_complete_pipeline_fails_when_no_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "pmarlo.markov_state_model._enhanced_impl.EnhancedMSM",
        _NoOpEnhancedMSM,
    )
    with pytest.raises(RuntimeError, match="No trajectory data were loaded"):
        run_complete_msm_analysis(
            trajectory_files=["missing.xtc"],
            topology_file=str(tmp_path / "structure.pdb"),
            output_dir=str(tmp_path),
            ignore_trajectory_errors=True,
        )


class _RecordingPipelineMSM:
    def __init__(self) -> None:
        self.built_lag: int | None = None
        self.built_method: str | None = None

    def build_msm(self, *, lag_time: int, method: str) -> None:
        self.built_lag = lag_time
        self.built_method = method

    def compute_implied_timescales(self) -> None:  # pragma: no cover - stub
        pass

    def generate_free_energy_surface(self, **_: object) -> None:  # pragma: no cover
        pass

    def create_state_table(self) -> None:  # pragma: no cover
        pass

    def extract_representative_structures(self) -> None:  # pragma: no cover
        pass

    def save_analysis_results(self) -> None:  # pragma: no cover
        pass

    def plot_free_energy_surface(self, **_: object) -> None:  # pragma: no cover
        pass

    def plot_implied_timescales(self, **_: object) -> None:  # pragma: no cover
        pass

    def plot_implied_rates(self, **_: object) -> None:  # pragma: no cover
        pass

    def plot_free_energy_profile(self, **_: object) -> None:  # pragma: no cover
        pass

    def plot_ck_test(self, **_: object) -> None:  # pragma: no cover
        pass


def test_build_and_analyze_accepts_numpy_temperatures() -> None:
    msm = _RecordingPipelineMSM()
    temps = np.array([300.0, 310.0])

    _build_and_analyze(msm, temperatures=temps, lag_time=7)

    assert msm.built_lag == 7
    assert msm.built_method == "tram"


def test_build_and_analyze_single_temperature_array_defaults_to_standard() -> None:
    msm = _RecordingPipelineMSM()
    temps = np.array([300.0])

    _build_and_analyze(msm, temperatures=temps, lag_time=5)

    assert msm.built_method == "standard"


def test_select_estimation_method_handles_numpy_sequences() -> None:
    temps = np.array([300.0, 310.0])
    assert _select_estimation_method(temps) == "tram"
