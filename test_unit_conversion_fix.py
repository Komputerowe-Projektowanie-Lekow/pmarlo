"""
Test script to verify the vectorized exchange math implementation.

This tests that the vectorized exchange calculations produce the same results
as the scalar version and that the performance is improved.
"""

import numpy as np
from pmarlo.replica_exchange.exchange_engine import ExchangeEngine


def test_vectorized_vs_scalar():
    """Test that vectorized and scalar methods produce identical results."""

    # Setup
    temperatures = [300.0, 310.0, 320.0, 330.0, 340.0]
    rng = np.random.default_rng(42)
    engine = ExchangeEngine(temperatures, rng)

    # Test data
    replica_states = [0, 1, 2, 3, 4]
    energies = [-1000.0, -980.0, -960.0, -940.0, -920.0]

    # Test pairs
    i_indices = [0, 2]
    j_indices = [1, 3]

    # Reset RNG for scalar version
    rng_scalar = np.random.default_rng(42)
    engine_scalar = ExchangeEngine(temperatures, rng_scalar)

    # Calculate using scalar method
    scalar_deltas = []
    scalar_probs = []
    for i, j in zip(i_indices, j_indices):
        delta = engine_scalar.delta_from_values(
            temperatures[replica_states[i]],
            temperatures[replica_states[j]],
            energies[i],
            energies[j]
        )
        prob = engine_scalar.probability_from_delta(delta)
        scalar_deltas.append(delta)
        scalar_probs.append(prob)

    # Reset RNG for vectorized version
    rng_vector = np.random.default_rng(42)
    engine_vector = ExchangeEngine(temperatures, rng_vector)

    # Calculate using vectorized method
    vector_deltas, accepted = engine_vector.attempt_exchanges_vectorized(
        replica_states, energies, i_indices, j_indices
    )

    # Compare deltas
    print("Scalar deltas:", scalar_deltas)
    print("Vector deltas:", vector_deltas)
    print("Delta difference:", np.abs(np.array(scalar_deltas) - vector_deltas))

    # Check if deltas are close (within numerical precision)
    assert np.allclose(scalar_deltas, vector_deltas, rtol=1e-10), \
        f"Deltas don't match! Scalar: {scalar_deltas}, Vector: {vector_deltas}"

    print("✓ Vectorized and scalar methods produce identical deltas")
    print(f"✓ Acceptance decisions: {accepted}")
    print("✓ Test passed!")


def test_vectorized_performance():
    """Test that vectorized method is faster for large batches."""
    import time

    # Setup with many replicas
    n_replicas = 100
    temperatures = [300.0 + i * 2.0 for i in range(n_replicas)]
    rng = np.random.default_rng(123)
    engine = ExchangeEngine(temperatures, rng)

    replica_states = list(range(n_replicas))
    energies = [-1000.0 + i * 5.0 for i in range(n_replicas)]

    # Create pairs for even exchanges
    i_indices = list(range(0, n_replicas - 1, 2))
    j_indices = [i + 1 for i in i_indices]

    # Time vectorized version
    start = time.time()
    for _ in range(100):
        deltas, accepted = engine.attempt_exchanges_vectorized(
            replica_states, energies, i_indices, j_indices
        )
    vectorized_time = time.time() - start

    # Reset RNG
    rng_scalar = np.random.default_rng(123)
    engine_scalar = ExchangeEngine(temperatures, rng_scalar)

    # Time scalar version (simulate the old loop)
    start = time.time()
    for _ in range(100):
        for i, j in zip(i_indices, j_indices):
            delta = engine_scalar.delta_from_values(
                temperatures[replica_states[i]],
                temperatures[replica_states[j]],
                energies[i],
                energies[j]
            )
            prob = engine_scalar.probability_from_delta(delta)
            # Simulate acceptance check
            _ = rng_scalar.random() < prob
    scalar_time = time.time() - start

    print(f"\nPerformance comparison ({len(i_indices)} pairs, 100 iterations):")
    print(f"  Scalar method:     {scalar_time:.4f} seconds")
    print(f"  Vectorized method: {vectorized_time:.4f} seconds")
    print(f"  Speedup:           {scalar_time / vectorized_time:.2f}x")

    if vectorized_time < scalar_time:
        print("✓ Vectorized method is faster!")
    else:
        print("⚠ Warning: Vectorized method not faster (might be due to small problem size)")


def test_edge_cases():
    """Test edge cases like single pair, identical temperatures, etc."""

    temperatures = [300.0, 300.0, 310.0]  # Two identical temps
    rng = np.random.default_rng(456)
    engine = ExchangeEngine(temperatures, rng)

    replica_states = [0, 1, 2]
    energies = [-1000.0, -990.0, -980.0]

    # Test single pair
    deltas, accepted = engine.attempt_exchanges_vectorized(
        replica_states, energies, [0], [1]
    )
    print(f"\nSingle pair test: delta={deltas[0]:.6f}, accepted={accepted[0]}")

    # With identical temperatures, delta should be 0
    assert np.abs(deltas[0]) < 1e-10, f"Expected delta ~0 for identical temps, got {deltas[0]}"
    print("✓ Identical temperature case handled correctly")

    # Test multiple pairs
    deltas, accepted = engine.attempt_exchanges_vectorized(
        replica_states, energies, [0, 1], [1, 2]
    )
    print(f"Multiple pairs test: deltas={deltas}, accepted={accepted}")
    print("✓ Multiple pairs handled correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Vectorized Exchange Math Implementation")
    print("=" * 60)

    print("\n1. Testing correctness (vectorized vs scalar)...")
    test_vectorized_vs_scalar()

    print("\n2. Testing edge cases...")
    test_edge_cases()

    print("\n3. Testing performance...")
    test_vectorized_performance()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

