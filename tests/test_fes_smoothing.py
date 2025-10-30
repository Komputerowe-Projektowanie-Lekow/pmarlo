import numpy as np

from pmarlo.markov_state_model.fes_smoothing import (
    adaptive_bandwidth,
    mark_bins_for_smoothing,
)


def test_uncertainty_decreases_with_samples():
    rng = np.random.default_rng(0)

    def sample_counts(N: int) -> np.ndarray:
        pts = rng.normal(size=(N, 2))
        H, _, _ = np.histogram2d(
            pts[:, 0],
            pts[:, 1],
            bins=64,
            range=[(-3, 3), (-3, 3)],
        )
        return H

    frac = []
    for N in (1_000, 10_000):
        counts = sample_counts(N)
        mask, _ = mark_bins_for_smoothing(
            counts,
            target_sd_kT=0.5,
            alpha=1e-6,
            kT=1.0,
        )
        frac.append(mask.mean())
    assert frac[1] < frac[0], "More data should reduce fraction of bins flagged"


def test_adaptive_bandwidth_scales_with_ess():
    ess = np.array([[1.0, 10.0], [100.0, 1000.0]])
    h = adaptive_bandwidth(ess, h0=1.0, ess_ref=10.0, h_min=0.2, h_max=5.0)
    assert h[0, 0] > h[0, 1] > h[1, 0] > h[1, 1]


def test_no_magic_thresholds_left():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    text = []
    for path in root.rglob("*.py"):
        if "build" in path.parts:
            continue
        try:
            text.append(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    blob = "\n".join(text)
    assert "0.3" not in blob or "empty bins" not in blob
    assert "0.4" not in blob or "force smoothing" not in blob
