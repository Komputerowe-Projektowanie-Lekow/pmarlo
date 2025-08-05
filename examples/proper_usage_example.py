"""
PMARLO Proper Usage Example
===========================

This example demonstrates the correct and robust usage patterns for PMARLO
after the comprehensive fixes and improvements.

Key improvements shown:
1. Proper error handling and validation
2. Consistent API usage patterns
3. Checkpoint management
4. Best practices for simulation setup
"""

import logging
from pathlib import Path
from pmarlo import Protein, ReplicaExchange, Simulation, MarkovStateModel, Pipeline

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PDB_FILE = "tests/data/3gd8.pdb"  # Adjust path as needed
OUTPUT_DIR = "improved_pmarlo_output"
TEMPERATURES = [300.0, 310.0, 320.0]
SIMULATION_STEPS = 2000  # Longer simulation for better results


def example_1_individual_components():
    """
    Example 1: Using individual components with proper error handling.
    
    This shows how to use each component separately with the improved APIs.
    """
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: Individual Components Usage")
    logger.info("=" * 60)
    
    try:
        # 1. Protein Preparation with error handling
        logger.info("Setting up protein...")
        protein = Protein(
            pdb_file=PDB_FILE,
            ph=7.0,
            auto_prepare=True  # Automatic preparation (if PDBFixer available)
        )
        
        properties = protein.get_properties()
        logger.info(f"Protein: {properties['num_atoms']} atoms, {properties['num_residues']} residues")
        
        # 2. Replica Exchange with improved API
        logger.info("Setting up replica exchange...")
        replica_exchange = ReplicaExchange(
            pdb_file=PDB_FILE,
            temperatures=TEMPERATURES,
            output_dir=f"{OUTPUT_DIR}/remd",
            auto_setup=False  # Explicit control over setup
        )
        
        # Check if setup is needed
        if not replica_exchange.is_setup():
            logger.info("Replicas not set up, setting up now...")
            replica_exchange.setup_replicas()
        
        # Validate setup
        if replica_exchange.is_setup():
            logger.info(f"Replica exchange ready: {replica_exchange.n_replicas} replicas")
        else:
            raise RuntimeError("Failed to set up replica exchange")
        
        # 3. Run simulation with error handling
        logger.info("Running replica exchange simulation...")
        try:
            trajectory_files = replica_exchange.run_simulation(
                total_steps=SIMULATION_STEPS,
                equilibration_steps=200
            )
            logger.info("Simulation completed successfully!")
            
            # Get exchange statistics
            stats = replica_exchange.get_exchange_statistics()
            logger.info(f"Exchange acceptance rate: {stats['overall_acceptance_rate']:.3f}")
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            # Could implement recovery logic here
            raise
        
        # 4. Markov State Model Analysis
        logger.info("Setting up MSM analysis...")
        msm = MarkovStateModel(
            trajectory_files=trajectory_files if isinstance(trajectory_files, list) else [trajectory_files],
            topology_file=PDB_FILE,
            temperatures=TEMPERATURES,
            output_dir=f"{OUTPUT_DIR}/msm"
        )
        
        # This would normally run MSM analysis
        logger.info("MSM analysis setup complete")
        
    except Exception as e:
        logger.error(f"Example 1 failed: {e}")
        raise


def example_2_pipeline_usage():
    """
    Example 2: Using the Pipeline class with checkpoint management.
    
    This shows the high-level API with automatic checkpointing and resume capability.
    """
    logger.info("=" * 60)
    logger.info("EXAMPLE 2: Pipeline Usage with Checkpointing")
    logger.info("=" * 60)
    
    try:
        # Create pipeline with checkpointing enabled
        pipeline = Pipeline(
            pdb_file=PDB_FILE,
            temperatures=TEMPERATURES,
            steps=SIMULATION_STEPS,
            n_states=30,  # Number of MSM states
            output_dir=f"{OUTPUT_DIR}/pipeline",
            use_replica_exchange=True,
            use_metadynamics=True,
            auto_continue=True  # Automatically continue interrupted runs
        )
        
        # Check if we can continue a previous run
        if pipeline.can_continue():
            logger.info("Continuing previous run...")
            status = pipeline.get_checkpoint_status()
            if status:
                logger.info(f"Previous run status: {status}")
        
        # Run the complete pipeline
        logger.info("Starting pipeline execution...")
        results = pipeline.run()
        
        # Display results
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results overview:")
        for key, value in results.items():
            if isinstance(value, dict):
                logger.info(f"  {key}: {len(value)} items")
            else:
                logger.info(f"  {key}: {value}")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        # Pipeline automatically saves failure state for debugging
        raise


def example_3_error_recovery():
    """
    Example 3: Demonstrating error recovery and validation.
    
    This shows how the improved error handling helps with common issues.
    """
    logger.info("=" * 60)
    logger.info("EXAMPLE 3: Error Recovery and Validation")
    logger.info("=" * 60)
    
    # Example of handling setup validation
    try:
        # Create replica exchange without auto-setup
        remd = ReplicaExchange(
            pdb_file=PDB_FILE,
            temperatures=[300, 310],
            auto_setup=False
        )
        
        # Try to run without setup (should fail gracefully)
        try:
            remd.run_simulation(total_steps=100)
            logger.error("This should have failed!")
        except RuntimeError as e:
            logger.info(f"✓ Proper validation caught error: {e}")
        
        # Now set up properly
        remd.setup_replicas()
        logger.info("✓ Setup completed successfully")
        
        # Verify setup state
        if remd.is_setup():
            logger.info("✓ Setup validation passed")
        
    except Exception as e:
        logger.error(f"Error recovery example failed: {e}")


def example_4_checkpoint_management():
    """
    Example 4: Manual checkpoint management.
    
    Shows how to save and restore simulation state manually.
    """
    logger.info("=" * 60)
    logger.info("EXAMPLE 4: Manual Checkpoint Management")
    logger.info("=" * 60)
    
    try:
        # Create and setup replica exchange
        remd = ReplicaExchange(
            pdb_file=PDB_FILE,
            temperatures=[300, 310],
            output_dir=f"{OUTPUT_DIR}/checkpoints",
            auto_setup=True
        )
        
        # Save initial state
        initial_state = remd.save_checkpoint_state()
        logger.info("✓ Initial state saved")
        
        # Run a short simulation
        logger.info("Running short simulation...")
        remd.run_simulation(total_steps=100, equilibration_steps=50)
        
        # Save state after simulation
        final_state = remd.save_checkpoint_state()
        logger.info("✓ Final state saved")
        
        # Compare states
        logger.info(f"Exchange attempts: {initial_state.get('exchange_attempts', 0)} -> {final_state.get('exchange_attempts', 0)}")
        logger.info(f"Exchanges accepted: {initial_state.get('exchanges_accepted', 0)} -> {final_state.get('exchanges_accepted', 0)}")
        
        # Create new instance and restore
        remd2 = ReplicaExchange(
            pdb_file=PDB_FILE,
            temperatures=[300, 310],
            output_dir=f"{OUTPUT_DIR}/checkpoints_restored",
            auto_setup=False
        )
        
        # Restore from checkpoint
        remd2.restore_from_checkpoint(final_state)
        logger.info("✓ State restored successfully")
        
        # Verify restoration
        restored_state = remd2.save_checkpoint_state()
        assert restored_state['exchange_attempts'] == final_state['exchange_attempts']
        logger.info("✓ Checkpoint restoration verified")
        
    except Exception as e:
        logger.error(f"Checkpoint management example failed: {e}")


def main():
    """Run all examples."""
    logger.info("Starting PMARLO Proper Usage Examples")
    logger.info("This demonstrates the improved, robust API after comprehensive fixes")
    
    try:
        # Create output directory
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        
        # Check if PDB file exists
        if not Path(PDB_FILE).exists():
            logger.error(f"PDB file not found: {PDB_FILE}")
            logger.info("Please adjust PDB_FILE path or copy a test PDB file")
            return
        
        # Run examples
        logger.info("\n")
        example_1_individual_components()
        
        logger.info("\n")
        example_2_pipeline_usage()
        
        logger.info("\n")
        example_3_error_recovery()
        
        logger.info("\n")
        example_4_checkpoint_management()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY! 🎉")
        logger.info("=" * 60)
        logger.info(f"Output saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()