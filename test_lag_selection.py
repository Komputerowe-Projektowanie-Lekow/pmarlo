#!/usr/bin/env python
"""Test script for automatic lag time selection from ITS."""

import numpy as np
import sys

def test_select_lag_from_its():
    """Test the select_lag_from_its function with synthetic data."""
    from pmarlo.markov_state_model._msm_utils import select_lag_from_its

    print("Testing select_lag_from_its function...")
    print("=" * 60)

    # Test 1: Clear plateau case
    print("\nTest 1: Clear plateau at lag=50")
    lags = np.array([1, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200])
    timescales = np.array([
        [10, 5, 2],
        [25, 12, 5],
        [45, 20, 8],
        [80, 35, 12],
        [95, 42, 15],
        [98, 45, 16],
        [100, 46, 17],  # Plateau starts here
        [101, 46.5, 17],
        [102, 47, 17.5],
        [103, 47.2, 18],
        [103.5, 47.5, 18],
        [104, 47.8, 18.2],
    ])

    selected_lag = select_lag_from_its(lags, timescales, min_lag_idx=3, plateau_threshold=0.15)
    print(f"✓ Selected lag: {selected_lag} frames")
    assert 40 <= selected_lag <= 80, f"Expected lag in [40, 80], got {selected_lag}"

    # Test 2: No clear plateau (should select from latter half with max timescale)
    print("\nTest 2: No clear plateau (monotonic increase)")
    lags2 = np.array([1, 5, 10, 20, 30, 40, 50, 60, 80, 100])
    timescales2 = np.array([
        [10, 5, 2],
        [25, 12, 5],
        [45, 20, 8],
        [80, 35, 12],
        [95, 42, 15],
        [110, 50, 18],
        [125, 58, 21],
        [140, 65, 24],
        [155, 72, 27],
        [170, 80, 30],
    ])

    selected_lag2 = select_lag_from_its(lags2, timescales2, min_lag_idx=3, plateau_threshold=0.15)
    print(f"✓ Selected lag: {selected_lag2} frames")
    assert selected_lag2 >= 50, f"Expected lag >= 50 for monotonic case, got {selected_lag2}"

    # Test 3: Empty input (should fallback to 10)
    print("\nTest 3: Empty input (fallback to default)")
    lags3 = np.array([])
    timescales3 = np.array([]).reshape(0, 3)

    selected_lag3 = select_lag_from_its(lags3, timescales3)
    print(f"✓ Selected lag: {selected_lag3} frames (fallback)")
    assert selected_lag3 == 10, f"Expected fallback to 10, got {selected_lag3}"

    # Test 4: NaN/invalid timescales
    print("\nTest 4: Data with NaN values")
    lags4 = np.array([1, 5, 10, 20, 30, 40, 50, 60])
    timescales4 = np.array([
        [np.nan, np.nan, np.nan],
        [25, 12, 5],
        [45, 20, 8],
        [80, 35, 12],
        [95, 42, 15],
        [98, 45, 16],
        [100, 46, 17],
        [101, 46.5, 17],
    ])

    selected_lag4 = select_lag_from_its(lags4, timescales4, min_lag_idx=1, plateau_threshold=0.15)
    print(f"✓ Selected lag: {selected_lag4} frames")
    assert selected_lag4 > 1, f"Expected lag > 1, got {selected_lag4}"

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    return True


def test_import():
    """Test that all modules can be imported."""
    print("\nTesting imports...")
    print("=" * 60)

    try:
        from pmarlo.markov_state_model._msm_utils import (
            select_lag_from_its,
            candidate_lag_ladder,
            build_simple_msm,
        )
        print("✓ Successfully imported from _msm_utils")

        from pmarlo.api.msm import analyze_msm
        print("✓ Successfully imported analyze_msm from api.msm")

        print("=" * 60)
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


if __name__ == "__main__":
    try:
        # Test imports first
        if not test_import():
            sys.exit(1)

        # Test the lag selection function
        if not test_select_lag_from_its():
            sys.exit(1)

        print("\n" + "🎉 " * 20)
        print("All validation tests passed successfully!")
        print("The automatic lag selection implementation is working correctly.")
        print("🎉 " * 20)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

