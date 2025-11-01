#!/usr/bin/env python
"""Quick REMD demo showing performance after platform selection fix.

This demonstrates that REMD now runs at expected speed after fixing the
Reference platform bug that was causing 6x slowdown.

Usage:
    python quick_remd_demo.py

Expected runtime on CPU: ~2-3 minutes for 1000 steps
"""

from __future__ import annotations

import time
from pathlib import Path

def main():
    print("=" * 80)
    print("QUICK REMD PERFORMANCE DEMO")
    print("=" * 80)
    print()
    print("This demo runs a short REMD simulation (1000 steps, 4 replicas)")
    print("to verify the platform selection fix is working.")
    print()
    print("Expected runtime: ~2-3 minutes on CPU")
    print("If it takes >10 minutes, the Reference platform bug is still present!")
    print()
    print("=" * 80)
    input("Press Enter to start...")

    # Find test PDB
    pdb_candidates = [
        Path(__file__).parent.parent / "tests" / "_assets" / "3gd8-fixed.pdb",
        Path(__file__).parent.parent / "tests" / "_assets" / "3gd8.pdb",
    ]
    pdb_file = None
    for candidate in pdb_candidates:
        if candidate.exists():
            pdb_file = candidate
            break

    if not pdb_file:
        print("ERROR: No test PDB file found!")
        return

    print(f"\nUsing PDB: {pdb_file}")
    print()

    # Import PMARLO API
    from pmarlo.api.replica_exchange import run_replica_exchange

    # Setup
    output_dir = Path("tmp_quick_demo")
    temperatures = [300.0, 310.0, 320.0, 330.0]
    total_steps = 1000

    print(f"Configuration:")
    print(f"  PDB: {pdb_file.name}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Total steps: {total_steps}")
    print(f"  Output: {output_dir}")
    print()

    # Run
    print("Starting REMD simulation...")
    print("-" * 80)
    start_time = time.time()

    try:
        traj_files, temps = run_replica_exchange(
            pdb_file=str(pdb_file),
            output_dir=str(output_dir),
            temperatures=temperatures,
            total_steps=total_steps,
            quick=True,
            random_seed=42,
        )

        elapsed = time.time() - start_time

        print("-" * 80)
        print()
        print("=" * 80)
        print("SIMULATION COMPLETE")
        print("=" * 80)
        print(f"Runtime: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
        print(f"Throughput: {total_steps * len(temperatures) / elapsed:.1f} total steps/sec")
        print()
        print(f"Generated {len(traj_files)} trajectory files:")
        for traj in traj_files:
            print(f"  - {Path(traj).name}")
        print()

        # Performance assessment
        steps_per_sec = total_steps * len(temperatures) / elapsed

        if steps_per_sec > 20:
            print("✅ PERFORMANCE: GOOD")
            print(f"   Throughput ({steps_per_sec:.1f} steps/s) is within expected range.")
            print("   Platform selection fix is working correctly!")
        elif steps_per_sec > 10:
            print("⚠️  PERFORMANCE: MARGINAL")
            print(f"   Throughput ({steps_per_sec:.1f} steps/s) is slower than expected.")
            print("   Check CPU load and available memory.")
        else:
            print("❌ PERFORMANCE: POOR")
            print(f"   Throughput ({steps_per_sec:.1f} steps/s) is very slow.")
            print("   The Reference platform bug may still be present!")
            print("   Check logs for 'Using Reference platform' message.")

        print()
        print("Cleanup:")
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"  Removed {output_dir}")

    except KeyboardInterrupt:
        print("\n\nSimulation cancelled by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 80)
    print("Demo complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
