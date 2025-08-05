"""
Checkpoint Usage Examples for PMARLO
=====================================

This script demonstrates how to use the enhanced checkpoint system in PMARLO
for both standalone applications and as a library/package.
"""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pmarlo import Pipeline, run_pmarlo
from pmarlo.manager import CheckpointManager

def example_1_auto_continue():
    """Example 1: Automatic continuation of interrupted runs."""
    print("="*60)
    print("EXAMPLE 1: Auto-Continue Functionality")
    print("="*60)
    
    # This automatically detects and continues any interrupted run
    # If no interrupted run exists, it starts a new one
    try:
        results = run_pmarlo(
            pdb_file="tests/data/3gd8.pdb",
            steps=1000,
            auto_continue=True,  # Key parameter for auto-resume
            output_dir="example_output"
        )
        
        print(f"Pipeline completed: {results['pipeline']['status']}")
        if 'checkpoint_id' in results['pipeline']:
            print(f"Checkpoint ID: {results['pipeline']['checkpoint_id']}")
            
    except Exception as e:
        print(f"Example 1 failed (expected if test files missing): {e}")

def example_2_programmatic_control():
    """Example 2: Programmatic checkpoint control."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Programmatic Checkpoint Control")
    print("="*60)
    
    try:
        # Create pipeline with checkpoint support
        pipeline = Pipeline(
            pdb_file="tests/data/3gd8.pdb",
            auto_continue=True,
            output_dir="example_output"
        )
        
        # Check if we can continue from previous run
        if pipeline.can_continue():
            status = pipeline.get_checkpoint_status()
            print(f"📁 Found existing run - Progress: {status['progress']}")
            print(f"🔄 Auto-continuing from: {status['current_stage']}")
        else:
            print("🆕 Starting new run")
        
        # Get checkpoint info
        if pipeline.checkpoint_manager:
            print(f"💾 Checkpoint ID: {pipeline.checkpoint_manager.run_id}")
            print(f"📂 Output directory: {pipeline.checkpoint_manager.run_dir}")
        
        # Would run the pipeline here:
        # results = pipeline.run()
        print("✅ Pipeline setup complete (run() commented out for demo)")
        
    except Exception as e:
        print(f"Example 2 failed (expected if test files missing): {e}")

def example_3_detect_interrupted():
    """Example 3: Auto-detect interrupted runs."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Auto-Detect Interrupted Runs")
    print("="*60)
    
    # Check for interrupted runs programmatically
    interrupted = CheckpointManager.auto_detect_interrupted_run("example_output")
    
    if interrupted:
        print(f"🔍 Found interrupted run: {interrupted.run_id}")
        summary = interrupted.get_run_summary()
        print(f"📊 Status: {summary['status']}")
        print(f"📈 Progress: {summary['progress']} ({summary['progress_percent']:.1f}%)")
        print(f"🎯 Current stage: {summary['current_stage']}")
        
        # Can continue this specific run
        try:
            config = interrupted.load_config()
            pdb_file = config.get('pdb_file', 'tests/data/3gd8.pdb')
            
            print(f"🔄 Continuing run {interrupted.run_id}...")
            # results = run_pmarlo(pdb_file, checkpoint_id=interrupted.run_id)
            print("✅ Would continue the interrupted run here")
            
        except Exception as e:
            print(f"Continue failed: {e}")
    else:
        print("🆕 No interrupted runs found")

def example_4_specific_checkpoint():
    """Example 4: Working with specific checkpoint IDs."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Specific Checkpoint ID Control")
    print("="*60)
    
    # Start with specific checkpoint ID
    checkpoint_id = "12345"
    
    try:
        # Create pipeline with specific ID
        pipeline = Pipeline(
            pdb_file="tests/data/3gd8.pdb",
            checkpoint_id=checkpoint_id,
            auto_continue=False,  # Disable auto-continue for explicit control
            output_dir="example_output"
        )
        
        print(f"💾 Using checkpoint ID: {checkpoint_id}")
        
        # Check status before running
        if pipeline.checkpoint_manager and pipeline.checkpoint_manager.can_continue():
            status = pipeline.get_checkpoint_status()
            print(f"📊 Previous run status: {status['status']}")
            if status['status'] == 'failed':
                print(f"❌ Previous run failed at: {status['current_stage']}")
                print("🔄 Will retry failed step...")
        
        # Would run the pipeline:
        # results = pipeline.run()
        print("✅ Pipeline configured with specific checkpoint")
        
    except Exception as e:
        print(f"Example 4 failed: {e}")

def example_5_library_integration():
    """Example 5: Library integration patterns."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Library Integration Patterns")
    print("="*60)
    
    # Pattern 1: Helper function for checking interrupted runs
    def check_and_continue_runs(output_dir="example_output"):
        """Helper to check and continue runs programmatically."""
        interrupted = CheckpointManager.auto_detect_interrupted_run(output_dir)
        
        if interrupted:
            return {
                'has_interrupted': True,
                'run_id': interrupted.run_id,
                'status': interrupted.get_run_summary()
            }
        return {'has_interrupted': False}
    
    # Pattern 2: Robust continuation function
    def continue_or_start_new(pdb_file, **kwargs):
        """Continue existing run or start new one."""
        try:
            # Try auto-continue first
            return run_pmarlo(pdb_file, auto_continue=True, **kwargs)
        except Exception as e:
            print(f"Auto-continue failed: {e}")
            # Fall back to new run
            return run_pmarlo(pdb_file, auto_continue=False, **kwargs)
    
    # Demonstrate usage
    status = check_and_continue_runs()
    print(f"🔍 Interrupted run check: {status}")
    
    # Would use in practice:
    # results = continue_or_start_new("tests/data/3gd8.pdb", steps=1000)
    print("✅ Library integration patterns demonstrated")

def example_6_different_pipeline_types():
    """Example 6: Checkpoints work with all pipeline types."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Different Pipeline Types")
    print("="*60)
    
    # REMD Pipeline with checkpoints
    print("🧬 REMD Pipeline with checkpoints:")
    try:
        remd_pipeline = Pipeline(
            pdb_file="tests/data/3gd8.pdb",
            use_replica_exchange=True,
            temperatures=[300, 310, 320],
            auto_continue=True,
            output_dir="remd_example"
        )
        
        if remd_pipeline.checkpoint_manager:
            print(f"   Steps: {remd_pipeline.checkpoint_manager.life_data['total_steps']}")
        print("   ✅ REMD pipeline with checkpoints configured")
    except Exception as e:
        print(f"   REMD pipeline failed: {e}")
    
    # Single simulation pipeline
    print("\n🔬 Single Simulation Pipeline with checkpoints:")
    try:
        sim_pipeline = Pipeline(
            pdb_file="tests/data/3gd8.pdb",
            use_replica_exchange=False,
            auto_continue=True,
            output_dir="sim_example"
        )
        
        if sim_pipeline.checkpoint_manager:
            print(f"   Steps: {sim_pipeline.checkpoint_manager.life_data['total_steps']}")
        print("   ✅ Single simulation pipeline with checkpoints configured")
    except Exception as e:
        print(f"   Single simulation pipeline failed: {e}")

def main():
    """Run all checkpoint examples."""
    print("PMARLO CHECKPOINT SYSTEM EXAMPLES")
    print("="*60)
    print("These examples show how to use checkpoints in different scenarios:")
    print("1. Auto-continue functionality")
    print("2. Programmatic control")
    print("3. Auto-detect interrupted runs")
    print("4. Specific checkpoint ID control")
    print("5. Library integration patterns")
    print("6. Different pipeline types")
    print()
    
    # Run examples
    example_1_auto_continue()
    example_2_programmatic_control()
    example_3_detect_interrupted()
    example_4_specific_checkpoint()
    example_5_library_integration()
    example_6_different_pipeline_types()
    
    print("\n" + "="*60)
    print("🎉 ALL CHECKPOINT EXAMPLES COMPLETED")
    print("="*60)
    print("\n💡 KEY TAKEAWAYS:")
    print("   ✅ Checkpoints now work with ALL pipeline types")
    print("   ✅ Auto-continue works seamlessly for library usage")
    print("   ✅ No user interaction required for programmatic use")
    print("   ✅ Flexible checkpoint management for different scenarios")
    print("   ✅ Backward compatible with legacy checkpoint system")

if __name__ == "__main__":
    main()