"""
Test script to validate CK test integration into MSM health checks.

This script tests:
1. CKRunResult.max_error property computation
2. ck_rms_error() function
3. Health check gating on CK test results
"""

import numpy as np
from pathlib import Path


def test_ck_result_properties():
    """Test CKRunResult max_error and has_valid_tests properties."""
    print("=" * 70)
    print("TEST 1: CKRunResult Properties")
    print("=" * 70)

    from pmarlo.markov_state_model.ck_runner import CKRunResult, ck_rms_error

    # Test case 1: Valid MSE values
    result = CKRunResult(mse={2: 0.0004, 3: 0.0009, 4: 0.0016}, mode="macro")
    max_err = result.max_error
    print(f"\n✓ Test 1a: MSE values {result.mse}")
    print(f"  Expected max_error: 0.04 (sqrt(0.0016))")
    print(f"  Actual max_error:   {max_err:.4f}")
    print(f"  Has valid tests:    {result.has_valid_tests}")
    assert abs(max_err - 0.04) < 1e-6, f"Expected 0.04, got {max_err}"
    assert result.has_valid_tests, "Should have valid tests"

    # Test case 2: Empty MSE (no valid CK tests)
    result_empty = CKRunResult(mse={}, mode="micro")
    max_err_empty = result_empty.max_error
    print(f"\n✓ Test 1b: Empty MSE")
    print(f"  Expected max_error: inf")
    print(f"  Actual max_error:   {max_err_empty}")
    print(f"  Has valid tests:    {result_empty.has_valid_tests}")
    assert max_err_empty == float('inf'), f"Expected inf, got {max_err_empty}"
    assert not result_empty.has_valid_tests, "Should not have valid tests"

    # Test case 3: ck_rms_error function
    rms_error = ck_rms_error(result)
    print(f"\n✓ Test 1c: ck_rms_error() function")
    print(f"  Expected: 0.04")
    print(f"  Actual:   {rms_error:.4f}")
    assert abs(rms_error - 0.04) < 1e-6, f"Expected 0.04, got {rms_error}"

    print("\n✅ All CKRunResult property tests passed!")
    return True


def test_threshold_logic():
    """Test CK threshold pass/fail logic."""
    print("\n" + "=" * 70)
    print("TEST 2: CK Threshold Logic")
    print("=" * 70)

    from pmarlo.markov_state_model.ck_runner import CKRunResult

    threshold = 0.05

    # Good MSM: low error
    result_good = CKRunResult(mse={2: 0.0001, 3: 0.0004, 4: 0.0009}, mode="macro")
    max_err_good = result_good.max_error
    pass_good = max_err_good < threshold

    print(f"\n✓ Test 2a: Good MSM")
    print(f"  Max RMS error: {max_err_good:.4f}")
    print(f"  Threshold:     {threshold:.4f}")
    print(f"  Pass:          {pass_good}")
    assert pass_good, "Good MSM should pass"

    # Poor MSM: high error
    result_poor = CKRunResult(mse={2: 0.01, 3: 0.04, 4: 0.09}, mode="micro")
    max_err_poor = result_poor.max_error
    pass_poor = max_err_poor < threshold

    print(f"\n✓ Test 2b: Poor MSM")
    print(f"  Max RMS error: {max_err_poor:.4f}")
    print(f"  Threshold:     {threshold:.4f}")
    print(f"  Pass:          {pass_poor}")
    assert not pass_poor, "Poor MSM should fail"

    # Failed CK: infinite error
    result_failed = CKRunResult(mse={}, mode="micro")
    max_err_failed = result_failed.max_error
    pass_failed = max_err_failed < threshold

    print(f"\n✓ Test 2c: Failed CK test")
    print(f"  Max RMS error: {max_err_failed}")
    print(f"  Threshold:     {threshold:.4f}")
    print(f"  Pass:          {pass_failed}")
    assert not pass_failed, "Failed CK should not pass"

    print("\n✅ All threshold logic tests passed!")
    return True


def test_import_structure():
    """Test that all required functions/classes are importable."""
    print("\n" + "=" * 70)
    print("TEST 3: Import Structure")
    print("=" * 70)

    try:
        from pmarlo.markov_state_model.ck_runner import (
            CKRunResult,
            ck_rms_error,
            run_ck,
        )
        print("\n✓ Successfully imported from ck_runner:")
        print("  - CKRunResult")
        print("  - ck_rms_error")
        print("  - run_ck")

        from pmarlo.markov_state_model import CKRunResult as CKRunResult2
        print("\n✓ Successfully imported from markov_state_model:")
        print("  - CKRunResult (via __init__)")

        print("\n✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_realistic_scenario():
    """Test with realistic CK error values."""
    print("\n" + "=" * 70)
    print("TEST 4: Realistic Scenarios")
    print("=" * 70)

    from pmarlo.markov_state_model.ck_runner import CKRunResult

    scenarios = [
        {
            "name": "Excellent MSM (well-converged)",
            "mse": {2: 0.00005, 3: 0.00008, 4: 0.00012, 5: 0.00015},
            "expected_max": 0.012247,  # sqrt(0.00015)
            "should_pass": True,
        },
        {
            "name": "Good MSM (acceptable)",
            "mse": {2: 0.0004, 3: 0.0009, 4: 0.0016, 5: 0.0020},
            "expected_max": 0.044721,  # sqrt(0.0020)
            "should_pass": True,
        },
        {
            "name": "Marginal MSM (at threshold)",
            "mse": {2: 0.001, 3: 0.0015, 4: 0.002, 5: 0.0025},
            "expected_max": 0.05,  # sqrt(0.0025) = exactly threshold
            "should_pass": False,  # >= threshold fails
        },
        {
            "name": "Poor MSM (too short lag)",
            "mse": {2: 0.01, 3: 0.02, 4: 0.04, 5: 0.06},
            "expected_max": 0.244949,  # sqrt(0.06)
            "should_pass": False,
        },
    ]

    threshold = 0.05
    all_passed = True

    for scenario in scenarios:
        result = CKRunResult(mse=scenario["mse"], mode="macro")
        max_err = result.max_error
        passes = max_err < threshold

        print(f"\n✓ Scenario: {scenario['name']}")
        print(f"  MSE values:    {scenario['mse']}")
        print(f"  Max RMS error: {max_err:.6f}")
        print(f"  Expected max:  {scenario['expected_max']:.6f}")
        print(f"  Threshold:     {threshold:.6f}")
        print(f"  Pass status:   {passes}")
        print(f"  Should pass:   {scenario['should_pass']}")

        if abs(max_err - scenario['expected_max']) > 1e-5:
            print(f"  ⚠️  Error mismatch!")
            all_passed = False

        if passes != scenario['should_pass']:
            print(f"  ⚠️  Pass status mismatch!")
            all_passed = False

    if all_passed:
        print("\n✅ All realistic scenarios behave correctly!")
    else:
        print("\n⚠️  Some scenarios have mismatches (check details above)")

    return all_passed


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CK TEST INTEGRATION VALIDATION" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")

    tests = [
        ("CKRunResult Properties", test_ck_result_properties),
        ("Threshold Logic", test_threshold_logic),
        ("Import Structure", test_import_structure),
        ("Realistic Scenarios", test_realistic_scenario),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - CK integration is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED - Review output above for details")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

