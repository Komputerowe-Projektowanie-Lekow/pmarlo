from simulation import prepare_system, production_run, feature_extraction, build_transition_model, relative_energies, plot_DG
from protein import Protein
from pathlib import Path

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

def main():
    #test_protein()
    controlMain()

if __name__ == "__main__":
    main()
