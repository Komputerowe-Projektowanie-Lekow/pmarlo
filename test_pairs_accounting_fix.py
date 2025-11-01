"""Test script to verify the pairs/frames accounting bug fix."""

import numpy as np
from pmarlo.analysis.debug_export import compute_analysis_debug
from pmarlo.analysis.counting import expected_pairs


def test_pairs_accounting_fix():
    """Test that total_frames_declared and expected_pairs are computed from actual dtraj lengths."""

    # Create test data with dtrajs that don't match shard metadata
    # Simulate the bug scenario: shard metadata says 0 frames, but dtrajs have actual data
    dtrajs = [
        np.array([0, 1, 2, 0, 1, 2, 0]),  # 7 frames
        np.array([1, 2, 0, 1, 2]),        # 5 frames
        np.array([0, 1, 2, 1, 0, 2, 1, 2, 0])  # 9 frames
    ]
    # Total: 7 + 5 + 9 = 21 frames

    # Create fake shard metadata with wrong lengths
    shards_meta = [
        {"id": "shard-0", "start": 0, "stop": 0, "length": 0, "temperature": 300.0},
        {"id": "shard-1", "start": 0, "stop": 0, "length": 0, "temperature": 300.0},
        {"id": "shard-2", "start": 0, "stop": 0, "length": 0, "temperature": 300.0},
    ]

    dataset = {
        "dtrajs": dtrajs,
        "__shards__": shards_meta
    }

    lag = 1
    count_mode = "sliding"

    # Run compute_analysis_debug
    result = compute_analysis_debug(dataset, lag=lag, count_mode=count_mode)

    # Check the fix: total_frames_declared should be sum of actual dtraj lengths
    expected_total_frames = sum(len(d) for d in dtrajs)
    actual_total_frames = result.summary["total_frames_declared"]

    print(f"Expected total_frames_declared: {expected_total_frames}")
    print(f"Actual total_frames_declared: {actual_total_frames}")

    assert actual_total_frames == expected_total_frames, \
        f"total_frames_declared mismatch: expected {expected_total_frames}, got {actual_total_frames}"

    # Check that expected_pairs is computed from actual dtraj lengths
    dtraj_lengths = [len(d) for d in dtrajs]
    stride = 1  # sliding mode
    expected_pair_count = expected_pairs(dtraj_lengths, lag, stride)
    actual_expected_pairs = result.summary["expected_pairs"]

    print(f"Expected expected_pairs: {expected_pair_count}")
    print(f"Actual expected_pairs: {actual_expected_pairs}")

    assert actual_expected_pairs == expected_pair_count, \
        f"expected_pairs mismatch: expected {expected_pair_count}, got {actual_expected_pairs}"

    # Check that counted_pairs matches expected_pairs (within tolerance)
    counted_pairs = result.summary["counted_pairs"]
    tolerance = len(dtrajs)  # one hop per segment

    print(f"Counted pairs: {counted_pairs}")
    print(f"Expected pairs: {expected_pair_count}")
    print(f"Tolerance: {tolerance}")

    assert abs(counted_pairs - expected_pair_count) <= tolerance, \
        f"Pair count mismatch beyond tolerance: counted {counted_pairs}, expected {expected_pair_count}, tolerance {tolerance}"

    # Check total_frames_with_states
    total_frames_with_states = result.summary["total_frames_with_states"]
    print(f"Total frames with states: {total_frames_with_states}")

    # Should match the sum of actual dtraj lengths (all states are valid >= 0)
    assert total_frames_with_states == expected_total_frames, \
        f"total_frames_with_states should equal total_frames_declared when all states are valid"

    print("\n✓ All checks passed! The pairs/frames accounting bug is fixed.")
    print(f"\nSummary:")
    print(f"  - total_frames_declared: {actual_total_frames} (from actual dtrajs, not shard metadata)")
    print(f"  - total_frames_with_states: {total_frames_with_states}")
    print(f"  - expected_pairs: {expected_pair_count}")
    print(f"  - counted_pairs: {counted_pairs}")
    print(f"  - difference: {abs(counted_pairs - expected_pair_count)} (within tolerance of {tolerance})")


if __name__ == "__main__":
    test_pairs_accounting_fix()

