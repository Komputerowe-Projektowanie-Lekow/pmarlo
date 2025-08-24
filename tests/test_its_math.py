import warnings

import matplotlib.pyplot as plt
import numpy as np

from pmarlo.markov_state_model import MarkovStateModel
from pmarlo.markov_state_model.utils import safe_timescales
from pmarlo.results import ITSResult


def test_safe_timescales_handles_invalid_eigenvalues():
    eigvals = np.array([0.9999999999, 0.9, -0.1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts = safe_timescales(10, eigvals)
    assert np.isfinite(ts[0])
    assert np.isfinite(ts[1])
    assert np.isnan(ts[2])


def test_plotting_with_nans(tmp_path):
    eig_samples = np.array([[0.9, -0.1], [0.8, -0.1]])
    ts_arr = safe_timescales(10, eig_samples)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ts_mean = np.nanmean(ts_arr, axis=0)
        ts_lo = np.nanpercentile(ts_arr, 5, axis=0)
        ts_hi = np.nanpercentile(ts_arr, 95, axis=0)

    msm = MarkovStateModel(output_dir=tmp_path)
    msm.implied_timescales = ITSResult(
        lag_times=np.array([10]),
        eigenvalues=np.array([[0.9, -0.1]]),
        eigenvalues_ci=np.zeros((1, 2, 2)),
        timescales=ts_mean[np.newaxis, :],
        timescales_ci=np.stack([ts_lo, ts_hi], axis=-1)[np.newaxis, :, :],
        rates=np.reciprocal(
            ts_mean, where=np.isfinite(ts_mean), out=np.full_like(ts_mean, np.nan)
        )[np.newaxis, :],
        rates_ci=np.zeros((1, 2, 2)),
        recommended_lag_window=None,
    )
    msm.time_per_frame_ps = 1.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        msm.plot_implied_timescales()
    legend_texts = [t.get_text() for t in plt.gca().get_legend().get_texts()]
    assert any(
        "NaNs indicate unstable eigenvalues at this τ" in t for t in legend_texts
    )
    plt.close()
