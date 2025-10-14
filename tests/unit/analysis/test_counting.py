from __future__ import annotations

from typing import List

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pmarlo.analysis.counting import expected_pairs


def _simulate_pairs(lengths: List[int], tau: int, strides: List[int]) -> int:
    total = 0
    for length, stride in zip(lengths, strides):
        length = max(0, int(length))
        stride = max(1, int(stride))
        for idx in range(0, max(0, length - tau), stride):
            if idx + tau < length:
                total += 1
    return total


@given(
    st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=5),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=1, max_value=10),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_expected_pairs_matches_simulation(
    lengths: List[int], tau: int, stride: int
) -> None:
    expected = expected_pairs(lengths, tau, stride)
    simulated = _simulate_pairs(lengths, tau, [stride] * len(lengths))
    assert expected == simulated


@given(
    st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=5),
    st.integers(min_value=0, max_value=10),
    st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=5),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_expected_pairs_supports_per_shard_stride(
    lengths: List[int],
    tau: int,
    strides: List[int],
) -> None:
    # Align strides list length with lengths by repeating last value as needed.
    if len(strides) < len(lengths):
        strides = strides + [strides[-1]] * (len(lengths) - len(strides))
    strides = strides[: len(lengths)]

    expected = expected_pairs(lengths, tau, strides)
    simulated = _simulate_pairs(lengths, tau, strides)
    assert expected == simulated
