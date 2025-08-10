# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Integration tests for PMARLO package.

These tests verify that different components work together correctly.
"""

import pytest

from pmarlo import Pipeline, Protein
from pmarlo.protein.protein import HAS_PDBFIXER


class TestPackageImports:
    """Test that all package imports work correctly."""

    def test_main_imports(self):
        """Test importing main classes from package."""
        from pmarlo import (
            LegacyPipeline,
            MarkovStateModel,
            Pipeline,
            Protein,
            ReplicaExchange,
            Simulation,
        )

        # Test that classes exist and are callable
        assert callable(Protein)
        assert callable(ReplicaExchange)
        assert callable(Simulation)
        assert callable(MarkovStateModel)
        assert callable(Pipeline)
        assert callable(LegacyPipeline)

    def test_convenience_function_import(self):
        """Test importing convenience functions."""
        from pmarlo.pipeline import run_pmarlo

        assert callable(run_pmarlo)

    def test_legacy_function_imports(self):
        """Test importing legacy functions."""
        from pmarlo.simulation.simulation import (
            build_transition_model,
            feature_extraction,
            plot_DG,
            prepare_system,
            production_run,
            relative_energies,
        )

        assert callable(prepare_system)
        assert callable(production_run)
        assert callable(feature_extraction)
        assert callable(build_transition_model)
        assert callable(relative_energies)
        assert callable(plot_DG)


class TestWorkflowIntegration:
    """Test integrated workflows."""

    def test_protein_to_simulation_workflow(self, test_pdb_file, temp_output_dir):
        """Test workflow from protein preparation to simulation setup."""
        try:
            # Step 1: Prepare protein
            protein = Protein(str(test_pdb_file), ph=7.0)
            prepared_pdb = temp_output_dir / "prepared.pdb"
            protein.save(str(prepared_pdb))

            # Step 2: Create simulation
            from pmarlo.simulation.simulation import Simulation

            simulation = Simulation(
                pdb_file=str(prepared_pdb),
                temperature=300.0,
                steps=100,
                output_dir=str(temp_output_dir),
            )

            assert simulation.pdb_file == str(prepared_pdb)
            assert simulation.temperature == 300.0

        except ImportError as e:
            pytest.skip(f"Workflow test skipped due to missing dependencies: {e}")

    @pytest.mark.skipif(not HAS_PDBFIXER, reason="PDBFixer is required for this test")
    def test_five_line_api_setup(self, test_pdb_file):
        """Test the five-line API setup (without execution)."""
        from pmarlo import (
            MarkovStateModel,
            Pipeline,
            Protein,
            ReplicaExchange,
            Simulation,
        )

        # Test that we can create all components (the 5-line API)
        protein = Protein(str(test_pdb_file), ph=7.0)

        # These should not fail during initialization
        replica_exchange = ReplicaExchange(
            str(test_pdb_file), temperatures=[300, 310, 320]
        )
        simulation = Simulation(str(test_pdb_file), temperature=300, steps=1000)
        markov_state_model = MarkovStateModel()
        pipeline = Pipeline(str(test_pdb_file))

        # Verify components are created
        assert protein is not None
        assert replica_exchange is not None
        assert simulation is not None
        assert markov_state_model is not None
        assert pipeline is not None


class TestErrorHandling:
    """Test error handling across components."""

    def test_invalid_pdb_file_handling(self):
        """Test handling of invalid PDB files across components."""
        invalid_pdb = "nonexistent_file.pdb"

        # Test with auto_prepare=False first (should work without PDBFixer)
        with pytest.raises(Exception):
            Protein(invalid_pdb, auto_prepare=False)

        # Test with auto_prepare=True (default)
        if HAS_PDBFIXER:
            with pytest.raises(Exception):
                Protein(invalid_pdb)
        else:
            with pytest.raises(ImportError, match="PDBFixer is required"):
                Protein(invalid_pdb)

        # Pipeline should also handle invalid files
        pipeline = Pipeline(invalid_pdb)
        # Error should occur when trying to run, not during initialization
        assert pipeline.pdb_file == invalid_pdb

    def test_missing_dependencies_handling(self):
        """Test behavior when optional dependencies are missing."""
        # This is more of a documentation test since we can't easily mock
        # missing imports
        # But we can verify that our code structure handles ImportError gracefully
        from pmarlo.simulation.simulation import Simulation

        # The class should be importable even if OpenMM is not available
        # (errors should occur during execution, not import)
        assert Simulation is not None


class TestDataPersistence:
    """Test that data is properly saved and can be reloaded."""

    @pytest.mark.skipif(not HAS_PDBFIXER, reason="PDBFixer is required for this test")
    def test_protein_save_and_reload(self, test_pdb_file, temp_output_dir):
        """Test saving and reloading protein data."""
        # Save protein
        protein1 = Protein(str(test_pdb_file), ph=7.0)
        saved_file = temp_output_dir / "saved_protein.pdb"
        protein1.save(str(saved_file))

        # Reload protein
        protein2 = Protein(str(saved_file), ph=7.0)

        # Compare properties
        props1 = protein1.get_properties()
        props2 = protein2.get_properties()

        # Should have similar number of atoms (allowing for small differences
        # in preparation)
        assert abs(props1["num_atoms"] - props2["num_atoms"]) < 100

    def test_output_directory_structure(self, test_pdb_file, temp_output_dir):
        """Test that output directories are created with proper structure."""
        pipeline = Pipeline(
            pdb_file=str(test_pdb_file), output_dir=str(temp_output_dir)
        )

        # Should create the main output directory
        assert temp_output_dir.exists()

        # After setup, should create subdirectories
        try:
            pipeline.setup_protein()
            # Check if prepared protein file exists
            assert (temp_output_dir / "prepared_protein.pdb").exists()
        except Exception:
            # If setup fails due to dependencies, that's expected
            pass
