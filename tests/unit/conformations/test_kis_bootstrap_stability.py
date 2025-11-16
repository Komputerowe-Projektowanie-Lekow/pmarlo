"""Behavior-focused bootstrap stability tests for KineticImportanceScore."""

import numpy as np
import pytest

from pmarlo.conformations.kinetic_importance import KineticImportanceScore


class DummyKISResult:
    """Minimal stand-in for ``KISResult`` used by the bootstrap helper."""

    def __init__(self, kis_scores, ranked_states):
        self.kis_scores = np.asarray(kis_scores, dtype=float)
        self.ranked_states = np.asarray(ranked_states, dtype=int)


def make_kis_with_stubs(monkeypatch, scenario: str) -> KineticImportanceScore:
    """Return a KineticImportanceScore instance with deterministic stubs applied."""
    T0 = np.zeros((3, 3))
    pi = np.ones(3) / 3.0
    kis = KineticImportanceScore(T0, pi)
    kis.n_states = 3

    def compute_stub(self, k_slow=None, its=None):
        T = np.asarray(getattr(self, "T"))
        if np.all(T == 0.0):
            kis_scores = np.array([3.0, 2.0, 1.0])
            ranked_states = np.array([0, 1, 2], dtype=int)
        elif np.all(T == 1.0):
            kis_scores = np.array([3.0, 1.0, 2.0])
            ranked_states = np.array([0, 2, 1], dtype=int)
        else:
            raise ValueError("Unexpected T in compute_stub")
        return DummyKISResult(kis_scores, ranked_states)

    if scenario == "all_identical":
        def rebuild_stub(self, dtrajs, lag=1):
            T_boot = np.zeros((3, 3))
            pi_boot = np.ones(3) / 3.0
            return T_boot, pi_boot
    elif scenario == "alternate_baseline_and_alternative":
        def rebuild_stub(self, dtrajs, lag=1):
            if not hasattr(self, "_boot_call"):
                self._boot_call = 0
            self._boot_call += 1

            if self._boot_call % 2 == 1:
                T_boot = np.zeros((3, 3))
            else:
                T_boot = np.ones((3, 3))
            pi_boot = np.ones(3) / 3.0
            return T_boot, pi_boot
    elif scenario == "always_fail":
        def rebuild_stub(self, dtrajs, lag=1):
            raise RuntimeError("Forced failure in _rebuild_msm")
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    monkeypatch.setattr(KineticImportanceScore, "compute", compute_stub, raising=False)
    monkeypatch.setattr(KineticImportanceScore, "_rebuild_msm", rebuild_stub, raising=False)
    return kis


def test_bootstrap_stability_identical_samples_give_full_stability(monkeypatch):
    """Identical samples keep top_n rankings fixed and KIS std hits zero."""
    kis = make_kis_with_stubs(monkeypatch, scenario="all_identical")
    dtrajs = [np.array([0, 1, 2, 1, 0], dtype=int)]

    stability, bootstrap_std = kis.bootstrap_stability(
        dtrajs=dtrajs,
        n_boot=20,
        k_slow=2,
        top_n=3,
    )

    assert stability == pytest.approx(1.0)
    assert bootstrap_std.shape == (3,)
    assert np.allclose(bootstrap_std, 0.0)


def test_bootstrap_stability_top1_robust_but_lower_states_vary(monkeypatch):
    """Top-1 membership stays stable although lower KIS components fluctuate."""
    kis = make_kis_with_stubs(monkeypatch, scenario="alternate_baseline_and_alternative")
    dtrajs = [np.array([0, 1, 2, 1, 0], dtype=int)]

    stability, bootstrap_std = kis.bootstrap_stability(
        dtrajs=dtrajs,
        n_boot=4,
        k_slow=2,
        top_n=1,
    )

    assert stability == pytest.approx(1.0)
    assert bootstrap_std.shape == (3,)
    assert bootstrap_std[0] == pytest.approx(0.0)
    assert bootstrap_std[1] == pytest.approx(0.5)
    assert bootstrap_std[2] == pytest.approx(0.5)


def test_bootstrap_stability_partial_overlap_top2(monkeypatch):
    """Alternating rankings yield a 0.75 top-2 stability metric."""
    kis = make_kis_with_stubs(monkeypatch, scenario="alternate_baseline_and_alternative")
    dtrajs = [np.array([0, 1, 2, 1, 0], dtype=int)]

    stability, bootstrap_std = kis.bootstrap_stability(
        dtrajs=dtrajs,
        n_boot=4,
        k_slow=2,
        top_n=2,
    )

    assert stability == pytest.approx(0.75)
    assert bootstrap_std.shape == (3,)
    assert bootstrap_std[0] == pytest.approx(0.0)
    assert bootstrap_std[1] == pytest.approx(0.5)
    assert bootstrap_std[2] == pytest.approx(0.5)


def test_bootstrap_stability_raises_if_all_samples_fail(monkeypatch):
    """Complete bootstrap failure should raise instead of returning fake stability."""
    kis = make_kis_with_stubs(monkeypatch, scenario="always_fail")
    dtrajs = [np.array([0, 1, 2, 1, 0], dtype=int)]

    with pytest.raises(RuntimeError):
        kis.bootstrap_stability(
            dtrajs=dtrajs,
            n_boot=5,
            k_slow=2,
            top_n=2,
        )
