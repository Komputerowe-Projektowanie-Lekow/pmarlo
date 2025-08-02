from simulation import prepare_system, production_run, feature_extraction, build_transition_model, relative_energies, plot_DG
from protein import Protein
from replica_exchange import ReplicaExchange, run_remd_simulation, setup_bias_variables
from enhanced_msm import EnhancedMSM, run_complete_msm_analysis
from checkpoint_manager import CheckpointManager, list_runs
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
TESTS_DIR = BASE_DIR / "tests"

def test_protein():
    pdb_file = TESTS_DIR / "3gd8.pdb"
    dcd_path = TESTS_DIR / "traj.dcd"
    pdb_fixed_path = TESTS_DIR / "3gd8-fixed.pdb"

    try:
        # Initialize and prepare the protein
        print("Initializing Protein...")
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

        print("Protein test completed successfully.")
    except Exception as e:
        print(f"An error occurred during the test: {e}")


def controlMain():
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

def run_remd_pipeline(run_id=None, continue_run=False, steps=1000, n_states=50):
    """Run the new REMD + Enhanced MSM pipeline with checkpoint support."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize checkpoint manager with correct output directory
    # Use parent directory of src/ to access the root-level output directory
    output_base_dir = Path(__file__).parent.parent / "output"
    
    if continue_run and run_id:
        # Load existing run
        try:
            checkpoint_manager = CheckpointManager.load_existing_run(run_id, str(output_base_dir))
            print(f"Resuming run {run_id}...")
            checkpoint_manager.print_status()
        except FileNotFoundError:
            print(f"Error: No existing run found with ID {run_id}")
            print(f"Looking in: {output_base_dir}")
            list_runs(str(output_base_dir))
            return
    elif continue_run and not run_id:
        # List available runs and ask user to specify
        print("Error: --continue requires --id to specify which run to continue")
        list_runs(str(output_base_dir))
        return
    else:
        # Start new run
        checkpoint_manager = CheckpointManager(run_id, str(output_base_dir))
        checkpoint_manager.setup_run_directory()
        print(f"Started new run with ID: {checkpoint_manager.run_id}")
    
    # File paths
    pdb_file = TESTS_DIR / "3gd8.pdb"
    pdb_fixed_path = checkpoint_manager.run_dir / "inputs" / "3gd8-fixed.pdb"
    remd_output_dir = checkpoint_manager.run_dir / "trajectories"
    msm_output_dir = checkpoint_manager.run_dir / "analysis"
    
    # Save configuration
    config = {
        "pdb_file": str(pdb_file),
        "steps": steps,
        "n_states": n_states,
        "temperatures": [300.0, 310.0, 320.0],  # 3 replicas with small 10K gaps for high exchange rates
        "use_metadynamics": True,
        "created_at": checkpoint_manager.life_data["created"]
    }
    checkpoint_manager.save_config(config)
    
    try:
        print("=" * 60)
        print("REPLICA EXCHANGE + ENHANCED MSM PIPELINE")
        print("=" * 60)

        # Use checkpoint manager to determine what to run next
        while True:
            next_step = checkpoint_manager.get_next_step()
            
            if next_step is None:
                print("\n🎉 All steps completed!")
                break
            
            # Clear failed status when retrying a step
            if next_step in [s.get("name") for s in checkpoint_manager.life_data["failed_steps"]]:
                print(f"\n🔄 Retrying failed step: {next_step}")
                checkpoint_manager.clear_failed_step(next_step)
            
            # Execute the appropriate step
            if next_step == "protein_preparation":
                checkpoint_manager.mark_step_started("protein_preparation")
                print("\n[Stage 1/6] Protein Preparation...")
                
                protein = Protein(str(pdb_file), ph=7.0)
                
                # Ensure the inputs directory exists before saving
                pdb_fixed_path.parent.mkdir(parents=True, exist_ok=True)
                
                protein.save(str(pdb_fixed_path))
                properties = protein.get_properties()
                print(f"Protein prepared: {properties['num_atoms']} atoms, {properties['num_residues']} residues")
                
                # Copy input files for reproducibility
                checkpoint_manager.copy_input_files([str(pdb_file)])
                
                checkpoint_manager.mark_step_completed("protein_preparation", {
                    "num_atoms": properties['num_atoms'],
                    "num_residues": properties['num_residues'],
                    "pdb_fixed_path": str(pdb_fixed_path)
                })
                
            elif next_step == "system_setup":
                checkpoint_manager.mark_step_started("system_setup")
                print("\n[Stage 2/6] System Setup...")
                print("Setting up temperature ladder for enhanced sampling...")
                
                # Just mark as completed - the actual setup happens in replica_initialization
                checkpoint_manager.mark_step_completed("system_setup", {
                    "temperatures": config["temperatures"],
                    "use_metadynamics": config["use_metadynamics"]
                })
                
            elif next_step in ["replica_initialization", "energy_minimization", "gradual_heating", 
                              "equilibration", "production_simulation", "trajectory_demux"]:
                print("\n[Stage 3/6] Replica Exchange Molecular Dynamics...")
                
                # Run REMD simulation with checkpoint integration
                demux_trajectory = run_remd_simulation(
                    pdb_file=str(pdb_fixed_path),
                    output_dir=str(remd_output_dir),
                    total_steps=steps,
                    temperatures=config["temperatures"],
                    use_metadynamics=config["use_metadynamics"],
                    checkpoint_manager=checkpoint_manager  # Pass checkpoint manager
                )
                
                print(f"REMD completed. Demultiplexed trajectory: {demux_trajectory}")
                # The REMD function handles its own checkpoints, so we continue the loop
                
            elif next_step == "trajectory_analysis":
                checkpoint_manager.mark_step_started("trajectory_analysis")
                print("\n[Stage 4/6] Enhanced Markov State Model Analysis...")
                
                # Reconstruct demux trajectory path
                demux_trajectory = str(remd_output_dir / "demuxed_trajectory.dcd")
                
                # Use demultiplexed trajectory if available, otherwise use all trajectories
                if demux_trajectory and Path(demux_trajectory).exists():
                    trajectory_files = [demux_trajectory]
                    analysis_temperatures = [300.0]  # Only target temperature
                else:
                    # Use all replica trajectories for TRAM analysis
                    trajectory_files = [str(remd_output_dir / f"replica_{i:02d}.dcd") for i in range(len(config["temperatures"]))]
                    trajectory_files = [f for f in trajectory_files if Path(f).exists()]
                    analysis_temperatures = config["temperatures"]

                if not trajectory_files:
                    raise ValueError("No trajectory files found for analysis")

                print(f"Analyzing {len(trajectory_files)} trajectories...")

                # Run complete MSM analysis
                msm = run_complete_msm_analysis(
                    trajectory_files=trajectory_files,
                    topology_file=str(pdb_fixed_path),
                    output_dir=str(msm_output_dir),
                    n_clusters=n_states,
                    lag_time=10,
                    feature_type="phi_psi",
                    temperatures=analysis_temperatures
                )
                
                checkpoint_manager.mark_step_completed("trajectory_analysis", {
                    "n_trajectories": len(trajectory_files),
                    "n_clusters": n_states,
                    "analysis_output": str(msm_output_dir)
                })
                
            else:
                print(f"Unknown step: {next_step}")
                break

        # Final summary
        print("\n[Stage 5/6] Pipeline Complete!")
        print(f"✓ Results saved to: {checkpoint_manager.run_dir}")
        print("✓ Ready for analysis and visualization")
        
        # Mark pipeline as completed
        checkpoint_manager.life_data["status"] = "completed"
        checkpoint_manager.save_life_data()
        
        # Print final status
        checkpoint_manager.print_status()
        
        return checkpoint_manager.run_dir
        
    except Exception as e:
        checkpoint_manager.mark_step_failed(checkpoint_manager.life_data["current_stage"], str(e))
        print(f"An error occurred in REMD pipeline: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheckpoint saved. You can resume with:")
        print(f"python main.py --mode remd --id {checkpoint_manager.run_id} --continue")
        return None


def run_comparison_analysis():
    """Run both pipelines and compare results."""
    print("=" * 60)
    print("PIPELINE COMPARISON")
    print("=" * 60)
    
    try:
        # Run original pipeline
        print("\n>>> Running Original Single-Temperature Pipeline...")
        controlMain()
        
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
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['original', 'remd', 'compare', 'test'],
        default='remd',
        help='Analysis mode to run (default: remd)'
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
        controlMain()
    elif args.mode == 'remd':
        run_remd_pipeline(
            run_id=args.id,
            continue_run=args.continue_run,
            steps=args.steps,
            n_states=args.states
        )
    elif args.mode == 'compare':
        run_comparison_analysis()


if __name__ == "__main__":
    main()
