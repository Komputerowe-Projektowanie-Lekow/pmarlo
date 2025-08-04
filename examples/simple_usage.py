"""
Simple usage example for PMARLO - OpenMM-like 5-line API

This demonstrates the simple interface for protein simulation and MSM analysis.
"""

from src import Protein, ReplicaExchange, Simulation, MarkovStateModel, Pipeline

# Method 1: Ultra-simple one-liner (what the user requested)
def one_liner_example():
    """Ultra-simple usage - everything in one line."""
    from src.pipeline import run_pmarlo
    
    results = run_pmarlo("tests/data/3gd8.pdb", temperatures=[300, 310, 320], steps=1000, n_states=50)
    print(f"Pipeline completed! Results in: {results['pipeline']['output_dir']}")


# Method 2: Five-line usage (as requested by user)
def five_line_example():
    """The requested 5-line usage pattern."""
    protein = Protein("tests/data/3gd8.pdb", ph=7.0)
    replica_exchange = ReplicaExchange("tests/data/3gd8-fixed.pdb", temperatures=[300, 310, 320])
    simulation = Simulation("tests/data/3gd8-fixed.pdb", temperature=300, steps=1000)
    markov_state_model = MarkovStateModel()
    
    # The orchestration method
    pipeline = Pipeline("tests/data/3gd8.pdb", use_replica_exchange=True, steps=1000)
    results = pipeline.run()


# Method 3: Component-by-component control
def detailed_example():
    """More control over individual components."""
    # Setup components
    protein = Protein("tests/data/3gd8.pdb", ph=7.0)
    prepared_pdb = "tests/data/3gd8-fixed.pdb"
    protein.save(prepared_pdb)
    
    # Choose simulation method
    use_replica_exchange = True
    
    if use_replica_exchange:
        # Replica exchange simulation
        remd = ReplicaExchange(
            pdb_file=prepared_pdb,
            temperatures=[300.0, 310.0, 320.0],
            output_dir="remd_output"
        )
        trajectories = remd.run_simulation(steps=1000)
    else:
        # Single simulation
        simulation = Simulation(
            pdb_file=prepared_pdb,
            temperature=300.0,
            steps=1000,
            use_metadynamics=True
        )
        trajectory_file, states = simulation.run_complete_simulation()
        trajectories = [trajectory_file]
    
    # MSM analysis
    msm = MarkovStateModel(output_dir="msm_analysis")
    if hasattr(msm, 'run_complete_analysis'):
        results = msm.run_complete_analysis(
            trajectory_files=trajectories,
            topology_file=prepared_pdb,
            n_clusters=50
        )
    
    print("Analysis complete!")


if __name__ == "__main__":
    print("PMARLO Usage Examples")
    print("=" * 50)
    
    print("\n1. Ultra-simple one-liner:")
    print("from src.pipeline import run_pmarlo")
    print("results = run_pmarlo('protein.pdb', temperatures=[300, 310, 320])")
    
    print("\n2. Five-line usage (as requested):")
    print("""
protein = Protein("protein.pdb", ph=7.0)
replica_exchange = ReplicaExchange("protein.pdb", temperatures=[300, 310, 320])
simulation = Simulation("protein.pdb", temperature=300, steps=1000)  
markov_state_model = MarkovStateModel()
pipeline = Pipeline("protein.pdb").run()  # Orchestrates everything
    """)
    
    print("\n3. Component-by-component control available for advanced users")
    
    # Uncomment to run actual examples (requires test files)
    # one_liner_example()
    # five_line_example()
    # detailed_example()