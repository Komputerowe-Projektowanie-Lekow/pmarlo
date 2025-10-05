#!/usr/bin/env python3
"""
PMARLO Pipeline Demonstration

This script demonstrates the main PMARLO API usage patterns including:
- Simple protein preparation
- REMD pipeline with checkpoints
- API comparison examples

For production use, see the other example programs.
"""

import sys
from pathlib import Path

# Add the src directory to Python path for development
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

try:
    from pmarlo import Pipeline, Protein, run_pmarlo
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure PMARLO is installed or run from the project root.")
    sys.exit(1)

# Test data paths
TESTS_DIR = BASE_DIR / "tests" / "data"


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
            output_dir=str(BASE_DIR / "output" / "demo"),
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
    auto_continue=True,
)

# Programmatic pipeline control
from pmarlo import Pipeline
pipeline = Pipeline("protein.pdb", auto_continue=True)
results = pipeline.run()

# Manual component setup
from pmarlo import Protein, ReplicaExchange, MarkovStateModel
protein = Protein("protein.pdb", ph=7.0)
pipeline = Pipeline("protein.pdb")
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
