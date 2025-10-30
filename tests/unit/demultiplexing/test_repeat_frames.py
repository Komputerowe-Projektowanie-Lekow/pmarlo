from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pmarlo.demultiplexing.demux_engine import _repeat_frames


def test_repeat_frames_creates_expected_stack():
    frame = np.arange(6, dtype=np.float32).reshape(2, 3)

    repeated = _repeat_frames(frame, 3)

    assert repeated.shape == (3, 2, 3)
    assert repeated.dtype == frame.dtype
    assert not np.shares_memory(repeated, frame)
    for idx in range(3):
        np.testing.assert_array_equal(repeated[idx], frame)


def test_repeat_frames_with_zero_count():
    frame = np.arange(6, dtype=np.float32).reshape(2, 3)

    repeated = _repeat_frames(frame, 0)

    assert repeated.shape == (0, 2, 3)
    assert repeated.dtype == frame.dtype
