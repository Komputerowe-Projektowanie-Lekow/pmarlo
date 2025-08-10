# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tests for the Protein class.
"""

from unittest.mock import patch

import pytest

from pmarlo.protein.protein import HAS_PDBFIXER, Protein


class TestProtein:
    """Test cases for Protein class."""

    def test_protein_initialization_without_pdbfixer(self, test_pdb_file):
        """Test protein initialization without PDBFixer."""
        with patch("pmarlo.protein.protein.HAS_PDBFIXER", False):
            protein = Protein(str(test_pdb_file), auto_prepare=False)
            assert protein.pdb_file == str(test_pdb_file)
            assert protein.ph == 7.0
            assert protein.fixer is None
            assert not protein.prepared

    def test_protein_properties_without_pdbfixer(self, test_pdb_file):
        """Test protein property access without PDBFixer."""
        with patch("pmarlo.protein.protein.HAS_PDBFIXER", False):
            protein = Protein(str(test_pdb_file), auto_prepare=False)
            properties = protein.get_properties()

            # Basic properties should be initialized to default values
            assert isinstance(properties, dict)
            assert all(
                key in properties
                for key in [
                    "num_atoms",
                    "num_residues",
                    "num_chains",
                    "molecular_weight",
                    "charge",
                    "logp",
                ]
            )
            # All values should be at their defaults since no preparation was done
            assert all(properties[key] == 0 for key in properties)

    def test_protein_save_without_pdbfixer(self, test_pdb_file, temp_output_dir):
        """Test that saving without PDBFixer raises appropriate error."""
        with (
            patch("pmarlo.protein.protein.HAS_PDBFIXER", False),
            patch("pmarlo.protein.protein.PDBFixer", None),
        ):
            protein = Protein(str(test_pdb_file), auto_prepare=False)
            output_file = temp_output_dir / "test_output.pdb"

            # Set prepared to True to trigger the save_prepared_pdb path
            protein.prepared = True

            with pytest.raises(
                ImportError, match="PDBFixer is required for saving prepared structures"
            ):
                protein.save(str(output_file))

    @pytest.mark.pdbfixer
    def test_protein_initialization(self, test_pdb_file):
        """Test protein initialization."""
        protein = Protein(str(test_pdb_file), ph=7.0)
        assert protein.pdb_file == str(test_pdb_file)
        assert protein.ph == 7.0
        assert protein.fixer is not None

    @pytest.mark.pdbfixer
    def test_protein_properties(self, test_pdb_file):
        """Test protein property calculation."""
        protein = Protein(str(test_pdb_file), ph=7.0)
        properties = protein.get_properties()

        assert "num_atoms" in properties
        assert "num_residues" in properties
        assert "num_chains" in properties
        assert properties["num_atoms"] > 0
        assert properties["num_residues"] > 0

    @pytest.mark.pdbfixer
    def test_protein_save(self, test_pdb_file, temp_output_dir):
        """Test protein saving functionality."""
        protein = Protein(str(test_pdb_file), ph=7.0)
        output_file = temp_output_dir / "test_output.pdb"

        protein.save(str(output_file))
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_protein_invalid_file(self):
        """Test protein initialization with invalid file."""
        # Test with auto_prepare=False first (should work without PDBFixer)
        with pytest.raises(Exception):
            Protein("nonexistent_file.pdb", auto_prepare=False)

        # Test with auto_prepare=True (default)
        if HAS_PDBFIXER:
            with pytest.raises(Exception):
                Protein("nonexistent_file.pdb")
        else:
            with pytest.raises(ImportError, match="PDBFixer is required"):
                Protein("nonexistent_file.pdb")

    def test_auto_prepare_flag(self, test_pdb_file):
        """Test auto_prepare flag behavior."""
        with patch("pmarlo.protein.protein.HAS_PDBFIXER", False):
            # With auto_prepare=False, initialization should work even without PDBFixer
            protein = Protein(str(test_pdb_file), auto_prepare=False)
            assert not protein.prepared
            assert protein.fixer is None

            # With auto_prepare=True (default)
            with pytest.raises(ImportError, match="PDBFixer is required"):
                Protein(str(test_pdb_file))

            # Manual preparation should respect PDBFixer availability
            protein = Protein(str(test_pdb_file), auto_prepare=False)
            with pytest.raises(ImportError, match="PDBFixer is required"):
                protein.prepare()

    def test_system_creation_without_pdbfixer(self, test_fixed_pdb_file):
        """Test system creation functionality without PDBFixer."""
        with patch("pmarlo.protein.protein.HAS_PDBFIXER", False):
            protein = Protein(str(test_fixed_pdb_file), auto_prepare=False)

            # Test default forcefield creation
            protein.create_system()
            system_info = protein.get_system_info()
            assert system_info["system_created"]
            assert system_info["num_forces"] > 0
            assert len(system_info["forces"]) > 0

            # Test custom forcefield creation
            custom_ff = ["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"]
            protein.create_system(forcefield_files=custom_ff)
            system_info = protein.get_system_info()
            assert system_info["system_created"]

    def test_system_info_without_system(self, test_pdb_file):
        """Test system info when no system is created."""
        with patch("pmarlo.protein.protein.HAS_PDBFIXER", False):
            protein = Protein(str(test_pdb_file), auto_prepare=False)
            system_info = protein.get_system_info()
            assert not system_info["system_created"]


class TestProteinIntegration:
    """Integration tests for Protein class."""

    def test_protein_workflow_without_pdbfixer(self, test_pdb_file, temp_output_dir):
        """Test basic protein workflow without PDBFixer."""
        with (
            patch("pmarlo.protein.protein.HAS_PDBFIXER", False),
            patch("pmarlo.protein.protein.PDBFixer", None),
        ):
            # Initialize protein without preparation
            protein = Protein(str(test_pdb_file), auto_prepare=False)
            assert not protein.prepared

            # Get properties (should be default values)
            properties = protein.get_properties()
            assert properties["num_atoms"] == 0

            # Verify that preparation-related operations raise appropriate errors
            with pytest.raises(ImportError, match="PDBFixer is required"):
                protein.prepare()

            # Set prepared to True to trigger the save_prepared_pdb path
            protein.prepared = True

            with pytest.raises(ImportError, match="PDBFixer is required"):
                output_file = temp_output_dir / "prepared_protein.pdb"
                protein.save(str(output_file))

    @pytest.mark.pdbfixer
    def test_protein_preparation_workflow(self, test_pdb_file, temp_output_dir):
        """Test complete protein preparation workflow."""
        # Initialize protein
        protein = Protein(str(test_pdb_file), ph=7.0)

        # Get properties
        properties = protein.get_properties()
        assert properties["num_atoms"] > 0

        # Save prepared protein
        output_file = temp_output_dir / "prepared_protein.pdb"
        protein.save(str(output_file))

        # Verify saved file
        assert output_file.exists()

        # Load saved protein and verify
        protein2 = Protein(str(output_file), ph=7.0)
        properties2 = protein2.get_properties()

        # Properties should be similar (allowing for small differences)
        assert abs(properties["num_atoms"] - properties2["num_atoms"]) < 100
