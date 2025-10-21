from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pmarlo.analysis.debug_export import total_pairs_from_shards


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(
    lengths=st.lists(st.integers(min_value=10, max_value=5000), min_size=2, max_size=6),
    tau=st.integers(min_value=1, max_value=1000),
    stride=st.integers(min_value=1, max_value=50),
)
def test_total_pairs_formula(lengths: list[int], tau: int, stride: int) -> None:
    effective_lengths = [1 + (length - 1) // stride for length in lengths]
    predicted = sum(max(0, length - tau) for length in effective_lengths)
    assert predicted == total_pairs_from_shards(effective_lengths, tau)
