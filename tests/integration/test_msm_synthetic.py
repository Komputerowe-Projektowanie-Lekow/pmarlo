from __future__ import annotations

import numpy as np

from pmarlo.analysis.debug_export import compute_analysis_debug


def simulate_two_well(
    n: int, tau_corr: int = 800, noise: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a 1D overdamped Langevin trajectory in a double-well potential."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    dt = 1.0
    for t in range(1, n):
        force = -4.0 * x[t - 1] * (x[t - 1] ** 2 - 1.0)
        x[t] = (
            x[t - 1]
            + (dt / tau_corr) * force
            + np.sqrt(2.0 * dt / tau_corr) * rng.normal(0.0, 1.0)
        )
    z = (x > 0.0).astype(np.int32)
    return x.reshape(-1, 1), z


def counts_from_labels(z: np.ndarray, tau: int) -> np.ndarray:
    C = np.zeros((2, 2), dtype=float)
    for t in range(0, len(z) - tau):
        C[int(z[t]), int(z[t + tau])] += 1.0
    return C


def diagonal_mass(counts: np.ndarray) -> float:
    if counts.size == 0:
        return float("nan")
    row_sum = counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    T = counts / row_sum
    return float(np.trace(T) / T.shape[0])


def _make_dataset(z: np.ndarray) -> dict[str, object]:
    length = int(z.shape[0])
    shard = {
        "id": "synthetic",
        "legacy_id": "synthetic",
        "start": 0,
        "stop": length,
        "length": length,
        "temperature": 300.0,
    }
    return {"__shards__": [shard], "dtrajs": [z.astype(int)]}


def test_diag_mass_decreases_with_lag():
    _, labels = simulate_two_well(20000, tau_corr=800, seed=42)
    taus = [100, 500, 1500]
    masses = [diagonal_mass(counts_from_labels(labels, tau)) for tau in taus]

    assert masses[0] > 0.6
    assert masses[0] > masses[1] > masses[2]
    assert masses[-1] < 0.7


def test_debug_export_counts_match_manual():
    _, labels = simulate_two_well(20000, tau_corr=800, seed=11)
    tau = 400
    dataset = _make_dataset(labels)

    debug = compute_analysis_debug(dataset, lag=tau, count_mode="sliding")
    expected = counts_from_labels(labels, tau)

    np.testing.assert_allclose(debug.counts, expected)
    assert int(debug.summary["total_pairs"]) == int(expected.sum())
