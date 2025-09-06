from __future__ import annotations

import pytest

from pmarlo.demultiplexing.demux_plan import build_demux_frame_windows


def assert_monotonic_half_open(windows: list[tuple[int, int]]):
    prev_stop = None
    for i, (start, stop) in enumerate(windows):
        assert start >= 0 and stop >= 0
        assert stop >= start, f"segment {i} empty or negative"
        if prev_stop is not None:
            assert (
                start >= prev_stop
            ), f"non-monotonic at segment {i}: start={start} < prev_stop={prev_stop}"
        prev_stop = stop


def test_5k_run_with_equilibration_and_stride1():
    total_md_steps = 4900
    equil_pre = 100
    equil_post = 100
    stride_steps = 1
    exchange_freq = 125
    # build windows should incorporate eq offset and produce:
    # [ [200,325), [325,450), [450,575), ... ]
    windows = build_demux_frame_windows(
        total_md_steps=total_md_steps,
        equilibration_steps_pre=equil_pre,
        equilibration_steps_post=equil_post,
        stride_steps=stride_steps,
        exchange_frequency_steps=exchange_freq,
        n_segments=None,  # infer from totals
    )
    assert_monotonic_half_open(windows)
    # Spot-check first three windows
    assert windows[0] == (200, 325)
    assert windows[1] == (325, 450)
    assert windows[2] == (450, 575)


def test_15k_run_with_equilibration_and_stride1():
    total_md_steps = 14900
    equil_pre = 100
    equil_post = 100
    stride_steps = 1
    exchange_freq = 375
    # Windows should be: [ [200,575), [575,950), ..., last partial up to 14900 )
    windows = build_demux_frame_windows(
        total_md_steps=total_md_steps,
        equilibration_steps_pre=equil_pre,
        equilibration_steps_post=equil_post,
        stride_steps=stride_steps,
        exchange_frequency_steps=exchange_freq,
        n_segments=None,
    )
    assert_monotonic_half_open(windows)
    assert windows[0] == (200, 575)
    # Last segment start should be 14825 with eq=200, exch=375
    assert windows[-1] == (14825, 14900)


@pytest.mark.parametrize("stride_steps,exchange_freq", [
    (4, 125),   # non-divisible pair
    (5, 375),   # non-divisible pair
])
def test_rounding_consistency_no_backtrack(stride_steps: int, exchange_freq: int):
    total_md_steps = 14900
    equil_pre = 100
    equil_post = 100
    windows = build_demux_frame_windows(
        total_md_steps=total_md_steps,
        equilibration_steps_pre=equil_pre,
        equilibration_steps_post=equil_post,
        stride_steps=stride_steps,
        exchange_frequency_steps=exchange_freq,
        n_segments=None,
    )
    assert_monotonic_half_open(windows)
    # Half-open: no duplicate boundary frames
    for i in range(1, len(windows)):
        assert windows[i - 1][1] <= windows[i][0]

