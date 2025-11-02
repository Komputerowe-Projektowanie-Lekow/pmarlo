import os
import random
import types

import pytest

pytest.importorskip("sklearn")
from sklearn.datasets import make_blobs

np = pytest.importorskip("numpy")

pytest.importorskip("torch")

from pmarlo.markov_state_model.clustering import cluster_microstates
from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM
from pmarlo.utils import seed
from pmarlo.utils.seed import set_global_seed


def _transition_from_labels(labels: np.ndarray, out_dir: str) -> np.ndarray:
    """Build a transition matrix for testing purposes."""
    msm = EnhancedMSM(random_state=None, output_dir=out_dir)
    msm.n_states = int(labels.max()) + 1 if labels.size else 0
    msm.dtrajs = [labels]
    msm._build_standard_msm(lag_time=1)
    assert msm.transition_matrix is not None
    return np.asarray(msm.transition_matrix, dtype=float)


def test_reproducible_clustering_and_msm(tmp_path):
    """Two runs with the same seed should be identical."""
    data, _ = make_blobs(n_samples=200, centers=4, n_features=2, random_state=0)

    set_global_seed(123)
    res1 = cluster_microstates(data, n_states=4, random_state=None)
    T1 = _transition_from_labels(res1.labels, str(tmp_path / "run1"))

    set_global_seed(123)
    res2 = cluster_microstates(data, n_states=4, random_state=None)
    T2 = _transition_from_labels(res2.labels, str(tmp_path / "run2"))

    assert np.array_equal(res1.labels, res2.labels)
    assert np.allclose(T1, T2)


def test_set_global_seed_synchronizes_python_random(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    set_global_seed(4321)
    first = random.random()

    set_global_seed(4321)
    second = random.random()

    assert first == second
    assert os.environ["PYTHONHASHSEED"] == str(4321 & 0xFFFFFFFF)


def test_set_global_seed_handles_legacy_torch(monkeypatch):
    class _CudaStub:
        def __init__(self) -> None:
            self.seed = None

        @staticmethod
        def is_available() -> bool:
            return True

        def manual_seed_all(self, seed: int) -> None:
            self.seed = seed

    class _CudnnStub:
        def __init__(self) -> None:
            self.deterministic = False
            self.benchmark = True

    class _TorchStub:
        def __init__(self) -> None:
            self.cuda = _CudaStub()
            self.backends = types.SimpleNamespace(cudnn=_CudnnStub())
            self.manual_seed_value = None
            self.set_deterministic_called_with: list[bool] = []

        def manual_seed(self, seed: int) -> None:
            self.manual_seed_value = seed

        def set_deterministic(self, flag: bool) -> None:
            self.set_deterministic_called_with.append(flag)

    stub = _TorchStub()
    monkeypatch.setattr(seed, "torch", stub)

    set_global_seed(7)

    assert stub.manual_seed_value == 7
    assert stub.cuda.seed == 7
    assert stub.set_deterministic_called_with == [True]
    assert stub.backends.cudnn.deterministic is True
    assert stub.backends.cudnn.benchmark is False
