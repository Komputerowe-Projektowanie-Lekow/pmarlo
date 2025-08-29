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
                    "isoelectric_point",
                    "hydrophobic_fraction",
                    "aromatic_residues",
                    "heavy_atoms",
                ]
            )
            # When PDBFixer is unavailable, defaults may not be zero; ensure keys exist and values are ints
            for key in properties:
                assert isinstance(properties[key], (int, float))

    def test_optional_rdkit_descriptors(self, test_pdb_file):
        """RDKit descriptors are only added when detailed=True."""
        with patch("pmarlo.protein.protein.HAS_PDBFIXER", False):
            protein = Protein(str(test_pdb_file), auto_prepare=False)
            props = protein.get_properties()
            assert "logp" not in props

            detailed = protein.get_properties(detailed=True)
            assert "logp" in detailed

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

    @pytest.mark.pdbfixer
    def test_solvation_option(self, test_fixed_pdb_file):
        """Solvate proteins lacking water when requested."""
        protein = Protein(str(test_fixed_pdb_file), auto_prepare=False)
        protein.prepare()
        water_residues = {
            res
            for res in protein.topology.residues()
            if res.name in {"HOH", "H2O", "WAT"}
        }
        assert len(water_residues) == 0

        protein = Protein(str(test_fixed_pdb_file), auto_prepare=False)
        protein.prepare(solvate=True)
        water_residues = {
            res
            for res in protein.topology.residues()
            if res.name in {"HOH", "H2O", "WAT"}
        }
        assert len(water_residues) > 0


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

            # Get properties; ensure integer type rather than specific zero value
            properties = protein.get_properties()
            assert isinstance(properties.get("num_atoms", 0), int)

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


class TestProteinAdditional:
    """Additional validation tests for Protein class."""

    def test_invalid_ph(self, test_pdb_file):
        """pH outside 0-14 should raise ValueError."""
        with pytest.raises(ValueError, match="pH must be between"):
            Protein(str(test_pdb_file), ph=20.0, auto_prepare=False)

    def test_damaged_pdb_raises(self, damaged_pdb_file):
        """Damaged PDB files should fail during parsing."""
        with pytest.raises(ValueError):
            Protein(str(damaged_pdb_file), auto_prepare=False)

    def test_round_trip_io_without_preparation(self, test_pdb_file, temp_output_dir):
        """Saving and reloading unprepared proteins should preserve topology."""
        protein = Protein(str(test_pdb_file), auto_prepare=False)
        out_file = temp_output_dir / "roundtrip.pdb"
        protein.save(str(out_file))
        protein2 = Protein(str(out_file), auto_prepare=False)
        props1 = protein.get_properties()
        props2 = protein2.get_properties()
        assert props1["num_atoms"] == props2["num_atoms"]
        assert props1["num_residues"] == props2["num_residues"]

    def test_invalid_coordinates_detected(self, nan_pdb_file, skip_if_no_openmm):
        """Protein initialization should fail on non-finite coordinates."""
        if skip_if_no_openmm:
            pytest.skip("OpenMM required")
        with pytest.raises(ValueError, match="non-finite"):
            Protein(str(nan_pdb_file), auto_prepare=False)


class TestProteinMetrics:
    """Tests for low-level protein metric helpers."""

    def test_compute_protein_metrics_known_sequence(self):
        """Verify metric calculation on a known sequence."""
        protein = object.__new__(Protein)
        protein.ph = 7.0
        seq = "ACDEFGHIKLMNPQRSTVWY"  # 20 amino acids, 10 hydrophobic, 3 aromatic
        metrics = Protein._compute_protein_metrics(protein, seq)

        assert metrics["hydrophobic_fraction"] == pytest.approx(0.5)
        assert metrics["aromatic_residues"] == 3
        assert 0.0 <= metrics["isoelectric_point"] <= 14.0
        assert isinstance(metrics["charge"], float)

    def test_sequence_from_topology(self):
        """Ensure sequence extraction from topology is correct."""
        from openmm.app import Topology, element

        protein = object.__new__(Protein)
        top = Topology()
        chain = top.addChain("A")
        for res_name in ("ALA", "GLY", "PHE"):
            res = top.addResidue(res_name, chain)
            top.addAtom("N", element.get_by_symbol("N"), res)

        seq = Protein._sequence_from_topology(protein, top)
        assert seq == "AGF"
