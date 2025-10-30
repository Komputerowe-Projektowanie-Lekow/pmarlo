#!/usr/bin/env python3
"""
PMARLO Pipeline Demonstration

This script demonstrates the main PMARLO API usage patterns including:
- Simple protein preparation
- REMD pipeline with checkpoints
- API comparison examples

For production use, see the other example programs.
"""

from pathlib import Path

from _example_support import assets_path, ensure_src_on_path

ensure_src_on_path()

from pmarlo import Pipeline, Protein, run_pmarlo

TESTS_DIR = assets_path()
OUTPUT_ROOT = Path(__file__).resolve().parent / "programs_outputs" / "demo_pipeline"


def test_protein():
    """Test protein preparation functionality."""
    pdb_file = TESTS_DIR / "3gd8.pdb"
    pdb_fixed_path = TESTS_DIR / "3gd8-fixed.pdb"

    try:
        print("Testing Protein class...")
        protein = Protein(str(pdb_file), ph=7.0)
        print("✓ Protein initialized successfully.")

        print("Saving prepared protein structure...")
        protein.save(str(pdb_fixed_path))
        print(f"✓ Prepared protein saved to: {pdb_fixed_path}")

        print("Retrieving protein properties...")
        properties = protein.get_properties()
        print("Protein properties:")
        for key, value in properties.items():
            print(f"  {key}: {value}")

        print("✓ Protein test completed successfully.")
    except Exception as e:
        print(f"✗ An error occurred during the test: {e}")


def run_simple_example():
    """Run a simple example with the new API."""
    try:
        print("Running simple PMARLO example...")

        pdb_file = str(TESTS_DIR / "3gd8.pdb")

        # Create a pipeline with minimal settings
        pipeline = Pipeline(
            pdb_file=pdb_file,
            steps=100,  # Very short for demo
            n_states=10,  # Fewer states for demo
            use_replica_exchange=False,  # Simpler for demo
            output_dir=str(OUTPUT_ROOT / "simple_run"),
            auto_continue=True,
        )

        print("🆕 Starting new run")

        # Demo the setup phases
        protein = pipeline.setup_protein()
        print(
            f"✓ Protein setup complete: {protein.get_properties()['num_atoms']} atoms"
        )

        simulation = pipeline.setup_simulation()
        print(f"✓ Simulation setup complete for {simulation.steps} steps")

        pipeline.setup_msm_analysis()
        print(f"✓ MSM setup complete for {pipeline.n_states} states")

        print("✓ Demo setup complete! To run full simulation, call pipeline.run()")

    except Exception as e:
        print(f"Demo failed (this is expected if test files are missing): {e}")


def demo_simple_api():
    """Demonstrate the simple API usage patterns."""
    print("=" * 60)
    print("PMARLO SIMPLE API DEMONSTRATION")
    print("=" * 60)

    print("\n🆕 NEW API Usage Patterns:")
    print("=" * 40)
    print(
        """
# Ultra-simple one-liner
from pmarlo import run_pmarlo
results = run_pmarlo(
    "protein.pdb",
    temperatures=[300, 310, 320],
    steps=1000,
    output_dir="/path/to/pmarlo/run",
    auto_continue=True,
)

# Programmatic pipeline control
from pmarlo import Pipeline
pipeline = Pipeline("protein.pdb", output_dir="/path/to/pmarlo/run", auto_continue=True)
results = pipeline.run()

# Manual component setup
from pmarlo import Protein, ReplicaExchange, MarkovStateModel
protein = Protein("protein.pdb", ph=7.0)
pipeline = Pipeline("protein.pdb", output_dir="/path/to/pmarlo/run")
results = pipeline.run()
    """
    )

    print("\n💡 API ADVANTAGES:")
    print("   ✅ Clean, simple interface")
    print("   ✅ Automatic checkpoint handling")
    print("   ✅ Library-friendly (no CLI dependencies)")
    print("   ✅ Consistent API across all features")
    print("   ✅ Better error handling and recovery")


def main():
    """Main demonstration function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="PMARLO Pipeline Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_pipeline.py --mode test      # Test protein preparation
  python demo_pipeline.py --mode simple    # Show API usage
  python demo_pipeline.py --mode demo      # Run simple demo
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["test", "simple", "demo"],
        default="simple",
        help="Demo mode to run (default: simple)",
    )

    args = parser.parse_args()

    print("PMARLO Pipeline Demo")
    print("https://github.com/pmarlo/pmarlo")
    print(f"Mode: {args.mode}")
    print()

    if args.mode == "test":
        test_protein()
    elif args.mode == "simple":
        demo_simple_api()
    elif args.mode == "demo":
        run_simple_example()


if __name__ == "__main__":
    main()
