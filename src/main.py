"""
PMARLO: Protein Markov State Model Analysis with Replica Exchange

Main entry point demonstrating both the legacy interface and the new clean API.
"""

# New clean API imports
from . import Protein, ReplicaExchange, Simulation, MarkovStateModel, Pipeline, LegacyPipeline
from .pipeline import run_pmarlo

# Legacy imports for compatibility
from .simulation.simulation import prepare_system, production_run, feature_extraction, build_transition_model, relative_energies, plot_DG
from .manager.checkpoint_manager import CheckpointManager, list_runs

from pathlib import Path
import argparse
import sys
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BASE_DIR / "tests" / "data"


def original_pipeline_with_dg():
    pdb_file = TESTS_DIR / "3gd8.pdb"
    dcd_path = TESTS_DIR / "traj.dcd"
    pdb_fixed_path = TESTS_DIR / "3gd8-fixed.pdb"

    try:
        # Initialize and prepare the protein
        print("Initializing Protein...")
        protein = Protein(str(pdb_file), ph=8.0)
        print("Protein initialized successfully.")

        # Save the prepared protein structure
        print("Saving prepared protein structure...")
        protein.save(str(pdb_fixed_path))
        print(f"Prepared protein saved to: {pdb_fixed_path}")

        # Get protein properties
        properties = protein.get_properties()
        print(f"Protein prepared: {properties['num_atoms']} atoms, {properties['num_residues']} residues")

        # Prepare system and metadynamics
        simulation, meta = prepare_system(str(pdb_fixed_path))  # Ensure absolute path is passed

        # Run production
        production_run(steps=None, simulation=simulation, meta=meta)

        # Feature extraction
        states = feature_extraction(str(dcd_path), str(pdb_fixed_path))  # Ensure absolute paths are passed

        # Build Markov model and print free energies
        DG = build_transition_model(states)
        print("Free energies (kcal/mol):", DG)
        plot_DG(DG)

        DG = relative_energies(DG)
        print("Relative energies (kcal/mol):", DG)
        plot_DG(DG)

    except Exception as e:
        print(f"An error occurred: {e}")


def test_protein():
    """Test protein preparation functionality."""
    pdb_file = TESTS_DIR / "3gd8.pdb"
    pdb_fixed_path = TESTS_DIR / "3gd8-fixed.pdb"

    try:
        # Initialize and prepare the protein
        print("Testing Protein class...")
        protein = Protein(str(pdb_file), ph=7.0)
        print("Protein initialized successfully.")

        # Save the prepared protein structure
        print("Saving prepared protein structure...")
        protein.save(str(pdb_fixed_path))
        print(f"Prepared protein saved to: {pdb_fixed_path}")

        # Get and print protein properties
        print("Retrieving protein properties...")
        properties = protein.get_properties()
        print("Protein properties:")
        for key, value in properties.items():
            print(f"  {key}: {value}")

        print("✓ Protein test completed successfully.")
    except Exception as e:
        print(f"✗ An error occurred during the test: {e}")


def run_remd_pipeline(run_id=None, continue_run=False, steps=1000, n_states=50):
    """Run the legacy REMD + Enhanced MSM pipeline with checkpoint support."""
    
    # Use the new LegacyPipeline class instead of the old function
    output_base_dir = Path(__file__).parent.parent / "output"
    pdb_file = TESTS_DIR / "3gd8.pdb"
    
    legacy_pipeline = LegacyPipeline(
        pdb_file=str(pdb_file),
        output_dir=str(output_base_dir),
        run_id=run_id,
        continue_run=continue_run
    )
    
    return legacy_pipeline.run_legacy_remd_pipeline(steps=steps, n_states=n_states)


def run_comparison_analysis():
    """Run both pipelines and compare results."""
    print("=" * 60)
    print("PIPELINE COMPARISON")
    print("=" * 60)
    
    try:
        # Run original pipeline
        print("\n>>> Running Original Single-Temperature Pipeline...")
        original_pipeline_with_dg()
        
        # Run new REMD pipeline
        print("\n>>> Running New REMD + Enhanced MSM Pipeline...")
        msm = run_remd_pipeline()
        
        if msm is not None:
            print("\n>>> Comparison Complete!")
            print("Check output directories for detailed results:")
            print(f"  - Original: {TESTS_DIR}")
            print(f"  - REMD: {BASE_DIR / 'remd_output'}")
            print(f"  - Enhanced MSM: {BASE_DIR / 'msm_analysis'}")
        
    except Exception as e:
        print(f"Error in comparison analysis: {e}")


def demo_simple_api():
    """Demonstrate the new simple 5-line API."""
    print("=" * 60)
    print("PMARLO SIMPLE API DEMONSTRATION")
    print("=" * 60)
    
    # The requested 5-line usage pattern
    print("Five-line usage (as requested):")
    print("""
protein = Protein("tests/data/3gd8.pdb", ph=7.0)
replica_exchange = ReplicaExchange("tests/data/3gd8-fixed.pdb", temperatures=[300, 310, 320])
simulation = Simulation("tests/data/3gd8-fixed.pdb", temperature=300, steps=1000)
markov_state_model = MarkovStateModel()
pipeline = Pipeline("tests/data/3gd8.pdb").run()  # Orchestrates everything
    """)
    
    # Demonstrate with actual code (commented out for safety)
    print("\nTo run this example, uncomment the following lines:")
    print("""
# protein = Protein(str(TESTS_DIR / "3gd8.pdb"), ph=7.0)
# pipeline = Pipeline(str(TESTS_DIR / "3gd8.pdb"), steps=100, use_replica_exchange=False)
# results = pipeline.run()
# print(f"Results saved to: {results['pipeline']['output_dir']}")
    """)
    
    print("\nUltra-simple one-liner:")
    print("results = run_pmarlo('protein.pdb', temperatures=[300, 310, 320], steps=1000)")


def run_simple_example():
    """Run a simple example with the new API (minimal steps for testing)."""
    try:
        print("Running simple PMARLO example...")
        
        # Use minimal parameters for quick testing
        pdb_file = str(TESTS_DIR / "3gd8.pdb")
        
        # Create a pipeline with minimal settings
        pipeline = Pipeline(
            pdb_file=pdb_file,
            steps=100,  # Very short for demo
            n_states=10,  # Fewer states for demo
            use_replica_exchange=False,  # Simpler for demo
            output_dir="demo_output"
        )
        
        # This would be the complete run
        # results = pipeline.run()
        
        # For now, just show the setup
        protein = pipeline.setup_protein()
        print(f"✓ Protein setup complete: {protein.get_properties()['num_atoms']} atoms")
        
        simulation = pipeline.setup_simulation()
        print(f"✓ Simulation setup complete for {simulation.steps} steps")
        
        msm = pipeline.setup_markov_state_model()
        print(f"✓ MSM setup complete for {pipeline.n_states} states")
        
        print("✓ Demo setup complete! To run full simulation, call pipeline.run()")
        
    except Exception as e:
        print(f"Demo failed (this is expected if test files are missing): {e}")


def demo_new_vs_legacy():
    """Demonstrate the difference between new and legacy APIs."""
    print("=" * 60)
    print("NEW API vs LEGACY API COMPARISON")
    print("=" * 60)
    
    print("\n🆕 NEW API (Recommended - Simple & Clean):")
    print("=" * 40)
    print("""
# Ultra-simple one-liner
from pmarlo import run_pmarlo
results = run_pmarlo("protein.pdb", temperatures=[300, 310, 320], steps=1000)

# Five-line usage
from pmarlo import Protein, ReplicaExchange, Simulation, MarkovStateModel, Pipeline
protein = Protein("protein.pdb", ph=7.0)
replica_exchange = ReplicaExchange("protein.pdb", temperatures=[300, 310, 320])  
simulation = Simulation("protein.pdb", temperature=300, steps=1000)
markov_state_model = MarkovStateModel()
pipeline = Pipeline("protein.pdb").run()  # Orchestrates everything

# Component-by-component control
pipeline = Pipeline("protein.pdb", temperatures=[300, 310, 320], steps=1000)
results = pipeline.run()
    """)
    
    print("\n🕐 LEGACY API (Advanced users, checkpointing):")
    print("=" * 40)
    print("""
# Legacy checkpoint-enabled pipeline  
from pmarlo import LegacyPipeline
legacy = LegacyPipeline("protein.pdb", run_id="12345", continue_run=True)
results = legacy.run_legacy_remd_pipeline(steps=1000, n_states=50)

# Original function-based approach
from pmarlo.simulation import prepare_system, production_run, feature_extraction
from pmarlo.protein import Protein
protein = Protein("protein.pdb")
protein.save("prepared.pdb")
simulation, meta = prepare_system("prepared.pdb")
production_run(1000, simulation, meta)
states = feature_extraction("traj.dcd", "prepared.pdb")
    """)
    
    print("\n💡 RECOMMENDATION: Use the new API unless you need specific legacy features!")
    print("   - Simpler and cleaner")
    print("   - Better error handling")
    print("   - More consistent interface")
    print("   - Easier to extend and maintain")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="PMARLO: Protein Markov State Model Analysis with Replica Exchange",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode original          # Run original single-T pipeline
  python main.py --mode remd             # Run new REMD + enhanced MSM pipeline  
  python main.py --mode compare          # Run both pipelines for comparison
  python main.py --mode test             # Test protein preparation only
  python main.py --mode simple           # Demonstrate new simple API
  python main.py --mode demo             # Run simple demo with new API
  python main.py --mode comparison       # Compare new vs legacy APIs
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['original', 'remd', 'compare', 'test', 'simple', 'demo', 'comparison'],
        default='simple',
        help='Analysis mode to run (default: simple)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=1000,
        help='Number of simulation steps (default: 1000 for fast testing)'
    )
    
    parser.add_argument(
        '--states',
        type=int,
        default=50,
        help='Number of MSM states (default: 50)'
    )
    
    parser.add_argument(
        '--id',
        type=str,
        help='5-digit run ID for checkpoint management'
    )
    
    parser.add_argument(
        '--continue',
        dest='continue_run',
        action='store_true',
        help='Continue from last successful checkpoint'
    )
    
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List all available runs and their status'
    )

    # Parse arguments, but handle the case where script is imported
    if len(sys.argv) == 1:
        # No arguments provided, run default
        args = parser.parse_args(['--mode', 'remd'])
    else:
        args = parser.parse_args()

    # Handle list-runs command
    if args.list_runs:
        list_runs()
        return

    print("PMARLO - Protein Markov State Model Analysis")
    print("https://github.com/yourusername/pmarlo")
    print(f"Mode: {args.mode}")
    print()

    if args.mode == 'test':
        test_protein()
    elif args.mode == 'original':
        original_pipeline_with_dg()
    elif args.mode == 'remd':
        run_remd_pipeline(
            run_id=args.id,
            continue_run=args.continue_run,
            steps=args.steps,
            n_states=args.states
        )
    elif args.mode == 'compare':
        run_comparison_analysis()
    elif args.mode == 'simple':
        demo_simple_api()
    elif args.mode == 'demo':
        run_simple_example()
    elif args.mode == 'comparison':
        demo_new_vs_legacy()


if __name__ == "__main__":
    main()
