"""
Tests for the Protein class.
"""

import pytest
from pathlib import Path
from src.protein.protein import Protein


class TestProtein:
    """Test cases for Protein class."""
    
    def test_protein_initialization(self, test_pdb_file):
        """Test protein initialization."""
        protein = Protein(str(test_pdb_file), ph=7.0)
        assert protein.pdb_file == str(test_pdb_file)
        assert protein.ph == 7.0
        assert protein.fixer is not None
    
    def test_protein_properties(self, test_pdb_file):
        """Test protein property calculation."""
        protein = Protein(str(test_pdb_file), ph=7.0)
        properties = protein.get_properties()
        
        assert 'num_atoms' in properties
        assert 'num_residues' in properties
        assert 'num_chains' in properties
        assert properties['num_atoms'] > 0
        assert properties['num_residues'] > 0
    
    def test_protein_save(self, test_pdb_file, temp_output_dir):
        """Test protein saving functionality."""
        protein = Protein(str(test_pdb_file), ph=7.0)
        output_file = temp_output_dir / "test_output.pdb"
        
        protein.save(str(output_file))
        assert output_file.exists()
        assert output_file.stat().st_size > 0
    
    def test_protein_invalid_file(self):
        """Test protein initialization with invalid file."""
        with pytest.raises(Exception):
            Protein("nonexistent_file.pdb")


class TestProteinIntegration:
    """Integration tests for Protein class."""
    
    def test_protein_preparation_workflow(self, test_pdb_file, temp_output_dir):
        """Test complete protein preparation workflow."""
        # Initialize protein
        protein = Protein(str(test_pdb_file), ph=7.0)
        
        # Get properties
        properties = protein.get_properties()
        assert properties['num_atoms'] > 0
        
        # Save prepared protein
        output_file = temp_output_dir / "prepared_protein.pdb"
        protein.save(str(output_file))
        
        # Verify saved file
        assert output_file.exists()
        
        # Load saved protein and verify
        protein2 = Protein(str(output_file), ph=7.0)
        properties2 = protein2.get_properties()
        
        # Properties should be similar (allowing for small differences due to preparation)
        assert abs(properties['num_atoms'] - properties2['num_atoms']) < 100  # Allow some flexibility