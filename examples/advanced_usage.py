"""
Advanced PMARLO Usage Examples

This file demonstrates advanced features like custom configurations,
checkpointing, and integration with other tools.
"""

from pathlib import Path
import sys

# Add src to path for examples
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src import Pipeline, LegacyPipeline


def example_custom_pipeline():
    """Example: Custom pipeline configuration."""
    print("=" * 60)
    print("ADVANCED EXAMPLE 1: Custom Pipeline Configuration")
    print("=" * 60)
    
    test_data = Path(__file__).parent.parent / "tests" / "data"
    pdb_file = test_data / "3gd8.pdb"
    
    if not pdb_file.exists():
        print(f"⚠️  Test file not found: {pdb_file}")
        return
    
    try:
        # Advanced configuration
        pipeline = Pipeline(
            pdb_file=str(pdb_file),
            temperatures=[295.0, 300.0, 305.0, 310.0, 315.0],  # 5 replicas
            n_replicas=5,
            steps=50000,  # Longer simulation
            n_states=100,  # More MSM states
            use_replica_exchange=True,
            use_metadynamics=True,
            output_dir="advanced_analysis",
            checkpoint_id="advanced_001"
        )
        
        print("✅ Advanced pipeline configured:")
        print(f"   🌡️  Temperatures: {pipeline.temperatures}")
        print(f"   📊 MSM states: {pipeline.n_states}")
        print(f"   🔄 Replica exchange: {pipeline.use_replica_exchange}")
        print(f"   ⚛️  Metadynamics: {pipeline.use_metadynamics}")
        
        print("\n💡 Run with: pipeline.run()")
        
    except Exception as e:
        print(f"❌ Advanced configuration failed: {e}")


def example_checkpointing():
    """Example: Using checkpointing and resuming runs."""
    print("=" * 60)
    print("ADVANCED EXAMPLE 2: Checkpointing and Resume")
    print("=" * 60)
    
    test_data = Path(__file__).parent.parent / "tests" / "data"
    pdb_file = test_data / "3gd8.pdb"
    
    if not pdb_file.exists():
        print(f"⚠️  Test file not found: {pdb_file}")
        return
    
    try:
        print("🔄 Starting new checkpointed run...")
        
        # Start a new run with checkpointing
        legacy_pipeline = LegacyPipeline(
            pdb_file=str(pdb_file),
            output_dir="checkpointed_runs",
            run_id="checkpoint_demo_001",
            continue_run=False  # Start new
        )
        
        print("✅ Checkpointed pipeline ready:")
        print(f"   📁 Output dir: {legacy_pipeline.output_base_dir}")
        print(f"   🆔 Run ID: {legacy_pipeline.run_id}")
        
        print("\n💡 To run: legacy_pipeline.run_legacy_remd_pipeline()")
        print("💡 To resume: LegacyPipeline(..., continue_run=True)")
        
        # Show how to resume
        print("\n🔄 To resume a failed/interrupted run:")
        resume_pipeline = LegacyPipeline(
            pdb_file=str(pdb_file),
            output_dir="checkpointed_runs",
            run_id="checkpoint_demo_001",
            continue_run=True  # Resume existing
        )
        print("✅ Resume pipeline configured")
        
    except Exception as e:
        print(f"❌ Checkpointing example failed: {e}")


def example_integration_workflow():
    """Example: Integration with analysis workflow."""
    print("=" * 60)
    print("ADVANCED EXAMPLE 3: Analysis Integration Workflow")
    print("=" * 60)
    
    test_data = Path(__file__).parent.parent / "tests" / "data"
    pdb_file = test_data / "3gd8.pdb"
    
    if not pdb_file.exists():
        print(f"⚠️  Test file not found: {pdb_file}")
        return
    
    try:
        print("🧬 Step 1: Protein Analysis")
        from src import Protein
        protein = Protein(str(pdb_file), ph=7.4)  # Physiological pH
        properties = protein.get_properties()
        
        print(f"   📊 Protein properties:")
        for key, value in properties.items():
            print(f"      {key}: {value}")
        
        print("\n⚛️  Step 2: Simulation Strategy Selection")
        # Choose strategy based on protein size
        if properties['num_atoms'] > 5000:
            print("   🔥 Large protein: Using replica exchange")
            use_remd = True
            temps = [298, 308, 318, 328]
        else:
            print("   🌡️  Small protein: Single temperature sufficient")
            use_remd = False
            temps = [300]
        
        print(f"\n🔄 Step 3: Pipeline Configuration")
        pipeline = Pipeline(
            pdb_file=str(pdb_file),
            temperatures=temps,
            use_replica_exchange=use_remd,
            steps=1000,  # Adjust based on system
            n_states=min(50, properties['num_residues']),  # Scale with size
            output_dir="integrated_analysis"
        )
        
        print(f"   ✅ Strategy: {'REMD' if use_remd else 'Single-T'}")
        print(f"   🌡️  Temperatures: {temps}")
        print(f"   📊 MSM states: {pipeline.n_states}")
        
        print("\n💡 Step 4: Ready for execution")
        print("   Run with: results = pipeline.run()")
        
    except Exception as e:
        print(f"❌ Integration workflow failed: {e}")


def example_performance_tuning():
    """Example: Performance tuning recommendations."""
    print("=" * 60)
    print("ADVANCED EXAMPLE 4: Performance Tuning")
    print("=" * 60)
    
    print("🚀 Performance tuning recommendations:")
    
    print("\n📏 System Size Recommendations:")
    print("   Small proteins (<1000 atoms):")
    print("     - Single temperature simulation")
    print("     - 10-50 MSM states")
    print("     - 10,000-100,000 steps")
    
    print("\n   Medium proteins (1000-5000 atoms):")
    print("     - 3-4 replica exchange temperatures")
    print("     - 50-100 MSM states")
    print("     - 100,000-1,000,000 steps")
    
    print("\n   Large proteins (>5000 atoms):")
    print("     - 4-8 replica exchange temperatures")
    print("     - 100-200 MSM states")
    print("     - 1,000,000+ steps")
    
    print("\n⚡ Hardware Optimization:")
    print("   CPU: Use Platform.getPlatformByName('CPU')")
    print("   GPU: Use Platform.getPlatformByName('CUDA') or 'OpenCL'")
    print("   Multiple GPUs: Use replica exchange across devices")
    
    print("\n💾 Memory Management:")
    print("   - Use checkpointing for long runs")
    print("   - Save trajectories periodically")
    print("   - Clean up intermediate files")
    
    print("\n🔧 Example optimized configuration:")
    try:
        # Example optimized setup (without execution)
        optimized_config = {
            "temperatures": [298.0, 305.0, 312.0, 320.0],  # Good exchange ratio
            "steps": 500000,  # Sufficient sampling
            "n_states": 75,  # Balanced resolution
            "use_replica_exchange": True,
            "use_metadynamics": True,
            "output_dir": "optimized_run"
        }
        
        print("   ✅ Optimized configuration ready")
        for key, value in optimized_config.items():
            print(f"      {key}: {value}")
            
    except Exception as e:
        print(f"❌ Performance tuning example failed: {e}")


def main():
    """Run all advanced examples."""
    print("🚀 PMARLO Advanced Usage Examples")
    print("=" * 80)
    
    print("\n🎓 Advanced features and best practices:")
    print("1. Custom pipeline configuration")
    print("2. Checkpointing and resuming runs")
    print("3. Integration workflow patterns")
    print("4. Performance tuning recommendations")
    
    print("\n⚠️  Note: These are configuration examples - actual execution")
    print("requires computational resources and proper dependencies.\n")
    
    # Run examples
    example_custom_pipeline()
    print("\n")
    
    example_checkpointing()
    print("\n")
    
    example_integration_workflow()
    print("\n")
    
    example_performance_tuning()
    
    print("\n" + "=" * 80)
    print("🎉 Advanced examples complete!")
    print("💡 Key takeaways:")
    print("   - Configure based on system size")
    print("   - Use checkpointing for long runs")
    print("   - Choose appropriate hardware")
    print("   - Monitor performance and adjust")


if __name__ == "__main__":
    main()