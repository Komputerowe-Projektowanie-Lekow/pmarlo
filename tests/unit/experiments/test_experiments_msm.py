import numpy as np

from pmarlo.experiments.msm import (
    _compute_ck_test_mse,
    _extract_dtrajs_and_frame_count,
)


class _DummyMSM:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_extract_dtrajs_handles_numpy_array():
    msm = _DummyMSM(dtrajs=np.array([[0, 1, 2], [2, 1, 0]]))

    flattened, total_frames = _extract_dtrajs_and_frame_count(msm)

    assert flattened == [0, 1, 2, 2, 1, 0]
    assert total_frames == 6


def test_compute_ck_mse_handles_numpy_array_input():
    msm = _DummyMSM(
        transition_matrix=np.array([[0.6, 0.4], [0.5, 0.5]]),
        dtrajs=np.array([[0, 1, 0, 1, 0, 1]]),
        n_states=2,
        lag_time=1,
    )

    mse = _compute_ck_test_mse(msm)

    assert isinstance(mse, float)
    assert mse >= 0.0
