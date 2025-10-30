import numpy as np

from pmarlo.validation.ck_rule import CKConfig, decide_ck


def _rngP(n: int, seed: int = 1) -> tuple[np.ndarray, np.random.Generator]:
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(n), size=n)
    return P, rng


def test_absolute_mode_replicates_legacy() -> None:
    n = 5
    P, rng = _rngP(n, seed=7)
    k = 3
    Pk_true = np.linalg.matrix_power(P, k)
    Pk_bad = np.clip(Pk_true + 0.3 * rng.normal(size=(n, n)), 1e-12, 1.0)
    Pk_bad /= Pk_bad.sum(axis=1, keepdims=True)
    counts = np.full(n, 1000)

    cfg = CKConfig(mode="absolute", absolute=0.15, min_pass_fraction=1.0, k_steps=(k,))
    dec = decide_ck({k: P}, {k: Pk_bad}, {k: counts}, cfg)
    assert not dec.passed


def test_ess_adjusted_catches_bias_beyond_noise() -> None:
    n = 6
    P, rng = _rngP(n, seed=11)
    k = 4
    Pk_true = np.linalg.matrix_power(P, k)
    counts = np.full(n, 2000)
    Pk_biased = np.clip(Pk_true + 0.02 * rng.normal(size=(n, n)), 1e-12, 1.0)
    Pk_biased /= Pk_biased.sum(axis=1, keepdims=True)

    cfg = CKConfig(
        mode="ess_adjusted",
        min_pass_fraction=1.0,
        per_lag_cap=0.25,
        k_steps=(k,),
        sigma_mult=3.0,
    )
    dec = decide_ck({k: P}, {k: Pk_biased}, {k: counts}, cfg)
    assert not dec.passed or dec.per_lag[k]["error"] > dec.per_lag[k]["threshold"]


def test_ess_adjusted_allows_within_noise() -> None:
    n = 6
    P, rng = _rngP(n, seed=19)
    k = 2
    Pk_true = np.linalg.matrix_power(P, k)
    counts = np.full(n, 5000)
    Pk_noisy = np.clip(Pk_true + 0.002 * rng.normal(size=(n, n)), 1e-12, 1.0)
    Pk_noisy /= Pk_noisy.sum(axis=1, keepdims=True)

    cfg = CKConfig(
        mode="ess_adjusted",
        min_pass_fraction=1.0,
        per_lag_cap=0.25,
        k_steps=(k,),
        sigma_mult=3.0,
    )
    dec = decide_ck({k: P}, {k: Pk_noisy}, {k: counts}, cfg)
    assert dec.passed
