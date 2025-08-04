"""
Basic PMARLO Usage Examples

This file demonstrates the different ways to use PMARLO for protein simulation
and Markov state model analysis.
"""

from pathlib import Path
import sys

# Try importing as installed package first, fallback to src for development
try:
    from pmarlo import Protein, Pipeline, run_pmarlo
except ImportError:
    # Development mode - add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from src import Protein, Pipeline
    from src.pipeline import run_pmarlo


def example_1_ultra_simple():
    """Example 1: Ultra-simple one-liner approach."""
    print("=" * 60)
    print("EXAMPLE 1: Ultra-Simple One-Liner")
    print("=" * 60)
    
    # Get test data path
    test_data = Path(__file__).parent.parent / "tests" / "data"
    pdb_file = test_data / "3gd8.pdb"
    
    if not pdb_file.exists():
        print(f"⚠️  Test file not found: {pdb_file}")
        print("Please ensure test data is available in tests/data/")
        return
    
    print("Running complete analysis in one line...")
    print(f"Input: {pdb_file}")
    
    try:
        # This does everything: protein prep, simulation, MSM analysis
        results = run_pmarlo(
            pdb_file=str(pdb_file),
            temperatures=[300, 310, 320],
            steps=100,  # Short for demo
            n_states=10,  # Few states for demo
            output_dir="example_1_output"
        )
        
        print("✅ Analysis completed!")
        print(f"📁 Results in: {results['pipeline']['output_dir']}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("This is expected if dependencies (OpenMM, etc.) are not installed")


def example_2_five_line_api():
    """Example 2: Five-line API as requested."""
    print("=" * 60)
    print("EXAMPLE 2: Five-Line API")
    print("=" * 60)
    
    try:
        from pmarlo import Protein, ReplicaExchange, Simulation, MarkovStateModel, Pipeline
    except ImportError:
        from src import Protein, ReplicaExchange, Simulation, MarkovStateModel, Pipeline
    
    # Get test data path
    test_data = Path(__file__).parent.parent / "tests" / "data"
    pdb_file = test_data / "3gd8.pdb"
    
    if not pdb_file.exists():
        print(f"⚠️  Test file not found: {pdb_file}")
        return
    
    print("Setting up components...")
    
    try:
        # The requested 5-line usage:
        protein = Protein(str(pdb_file), ph=7.0)
        replica_exchange = ReplicaExchange(str(pdb_file), temperatures=[300, 310, 320])
        simulation = Simulation(str(pdb_file), temperature=300, steps=100)
        markov_state_model = MarkovStateModel()
        pipeline = Pipeline(str(pdb_file), output_dir="example_2_output")
        
        print("✅ All components created successfully!")
        print("💡 To run the full analysis: results = pipeline.run()")
        
        # Show component details
        print(f"🧬 Protein: {protein.pdb_file}")
        print(f"🔥 Replica Exchange: {len(replica_exchange.temperatures)} temperatures")
        print(f"⚛️  Simulation: {simulation.steps} steps at {simulation.temperature}K")
        print(f"📊 MSM: Ready for analysis")
        print(f"🔄 Pipeline: Orchestrates all components")
        
    except Exception as e:
        print(f"❌ Component setup failed: {e}")
        print("This is expected if dependencies are not installed")


def example_3_component_by_component():
    """Example 3: Component-by-component control."""
    print("=" * 60)
    print("EXAMPLE 3: Component-by-Component Control")
    print("=" * 60)
    
    # Get test data path
    test_data = Path(__file__).parent.parent / "tests" / "data"
    pdb_file = test_data / "3gd8.pdb"
    
    if not pdb_file.exists():
        print(f"⚠️  Test file not found: {pdb_file}")
        return
    
    try:
        print("Step 1: Protein Preparation")
        protein = Protein(str(pdb_file), ph=7.0)
        prepared_pdb = "example_3_prepared.pdb"
        protein.save(prepared_pdb)
        properties = protein.get_properties()
        print(f"✅ Protein prepared: {properties['num_atoms']} atoms, {properties['num_residues']} residues")
        
        print("\nStep 2: Create Pipeline with Custom Settings")
        pipeline = Pipeline(
            pdb_file=prepared_pdb,
            temperatures=[298.0, 308.0, 318.0],  # Custom temperatures
            steps=100,  # Short for demo
            n_states=15,  # Custom number of states
            use_replica_exchange=True,
            use_metadynamics=True,
            output_dir="example_3_output"
        )
        print("✅ Pipeline configured")
        
        print("\nStep 3: Setup Individual Components")
        components = pipeline.get_components()
        print(f"📦 Available components: {list(components.keys())}")
        
        print("\n💡 To run full analysis: pipeline.run()")
        print("💡 To run individual steps: pipeline.setup_protein(), etc.")
        
    except Exception as e:
        print(f"❌ Component control failed: {e}")


def example_4_legacy_compatibility():
    """Example 4: Legacy API compatibility."""
    print("=" * 60)
    print("EXAMPLE 4: Legacy API Compatibility")
    print("=" * 60)
    
    print("The old function-based approach still works...")
    
    try:
        try:
            from pmarlo.simulation.simulation import prepare_system, production_run, feature_extraction
            from pmarlo.simulation.simulation import build_transition_model, relative_energies, plot_DG
        except ImportError:
            from src.simulation.simulation import prepare_system, production_run, feature_extraction
            from src.simulation.simulation import build_transition_model, relative_energies, plot_DG
        
        print("✅ Legacy functions available:")
        print("  - prepare_system()")
        print("  - production_run()")
        print("  - feature_extraction()")
        print("  - build_transition_model()")
        print("  - relative_energies()")
        print("  - plot_DG()")
        
        print("\n🔧 Advanced users can still use checkpointed runs:")
        try:
            from pmarlo import LegacyPipeline
        except ImportError:
            from src import LegacyPipeline
        
        test_data = Path(__file__).parent.parent / "tests" / "data"
        pdb_file = test_data / "3gd8.pdb"
        
        if pdb_file.exists():
            legacy = LegacyPipeline(
                pdb_file=str(pdb_file),
                output_dir="example_4_output",
                run_id="demo123"
            )
            print("✅ Legacy pipeline available for advanced checkpointing")
        
    except Exception as e:
        print(f"❌ Legacy compatibility check failed: {e}")


def main():
    """Run all examples."""
    print("🚀 PMARLO Usage Examples")
    print("=" * 80)
    
    print("\n📚 This demo shows different ways to use PMARLO:")
    print("1. Ultra-simple one-liner")
    print("2. Five-line API (as requested)")
    print("3. Component-by-component control")
    print("4. Legacy compatibility")
    
    print("\n⚠️  Note: Some examples may fail if OpenMM/dependencies are not installed")
    print("This is expected and demonstrates graceful error handling.\n")
    
    # Run examples
    example_1_ultra_simple()
    print("\n")
    
    example_2_five_line_api()
    print("\n")
    
    example_3_component_by_component()
    print("\n")
    
    example_4_legacy_compatibility()
    
    print("\n" + "=" * 80)
    print("🎉 Examples complete!")
    print("💡 Choose the approach that best fits your needs:")
    print("   - One-liner for simplicity")
    print("   - Five-line API for OpenMM-like usage")
    print("   - Component control for customization")
    print("   - Legacy API for advanced features")


if __name__ == "__main__":
    main()