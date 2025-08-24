import numpy as np

from pmarlo.markov_state_model.enhanced_msm import EnhancedMSM


def test_bootstrap_counts_consistency(monkeypatch):
    msm = EnhancedMSM()
    msm.dtrajs = [np.array([0, 1, 0, 2, 1, 0])]
    msm.n_states = 3
    assignments = np.concatenate(msm.dtrajs)
    counts = np.bincount(assignments, minlength=msm.n_states)
    rng = np.random.default_rng(0)
    monkeypatch.setattr(np.random, "default_rng", lambda: rng)
    samples = msm._bootstrap_counts(assignments, n_boot=500)
    mean_counts = samples.mean(axis=0)
    assert np.allclose(mean_counts, counts, atol=0.1 * counts.max())
