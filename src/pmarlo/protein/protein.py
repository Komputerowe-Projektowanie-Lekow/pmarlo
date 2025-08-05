# PDBFixer is optional - users can install with: pip install "pmarlo[fixer]"
try:
    from pdbfixer import PDBFixer
    HAS_PDBFIXER = True
except ImportError:
    PDBFixer = None
    HAS_PDBFIXER = False
from openmm.app import PDBFile, ForceField
from openmm import unit
import os
from typing import Optional, Dict, Any, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt


class Protein:
    def __init__(self, pdb_file: str,
                 ph: float = 7.0,
                 auto_prepare: bool = True,
                 preparation_options: Optional[Dict[str, Any]] = None):
        """
        Initialize a Protein object with a PDB file.
        
        Args:
            pdb_file (str): Path to the PDB file
            ph (float): pH value for protonation state (default: 7.0)
            auto_prepare (bool): Automatically prepare the protein (default: True)
            preparation_options (Optional[Dict]): Custom preparation options
        """
        self.pdb_file = pdb_file
        self.ph = ph
        
        # PDBFixer object for protein preparation
        if not HAS_PDBFIXER:
            self.fixer = None
            self.prepared = False
            # If auto_prepare is True but PDBFixer is not available, raise an error
            if auto_prepare:
                raise ImportError(
                    "PDBFixer is required for protein preparation but is not installed. "
                    "Install it with: pip install 'pmarlo[fixer]' "
                    "or set auto_prepare=False to skip preparation."
                )
        else:
            self.fixer = PDBFixer(filename=pdb_file)
            self.prepared = False
        
        # Store protein data
        self.topology = None
        self.positions = None
        self.forcefield = None
        self.system = None
        
        # RDKit molecule object for property calculations
        self.rdkit_mol = None
        
        # Protein properties
        self.properties = {
            'num_atoms': 0,
            'num_residues': 0,
            'num_chains': 0,
            'molecular_weight': 0.0,
            'exact_molecular_weight': 0.0,
            'charge': 0.0,
            'logp': 0.0,
            'hbd': 0,  # Hydrogen bond donors
            'hba': 0,  # Hydrogen bond acceptors
            'rotatable_bonds': 0,
            'aromatic_rings': 0,
            'heavy_atoms': 0
        }
        
        if auto_prepare:
            prep_options = preparation_options or {}
            prep_options.setdefault('ph', ph)
            self.prepare(**prep_options)
    
    def prepare(self, 
                ph: float = 7.0,
                remove_heterogens: bool = True,
                keep_water: bool = False,
                add_missing_atoms: bool = True,
                add_missing_hydrogens: bool = True,
                replace_nonstandard_residues: bool = True,
                find_missing_residues: bool = True,
                **kwargs) -> 'Protein':
        """
        Prepare the protein structure with specified options.
        
        Args:
            ph (float): pH value for protonation state (default: 7.0)
            remove_heterogens (bool): Remove non-protein molecules (default: True)
            keep_water (bool): Keep water molecules if True (default: False)
            add_missing_atoms (bool): Add missing atoms to residues (default: True)
            add_missing_hydrogens (bool): Add missing hydrogens (default: True)
            replace_nonstandard_residues (bool): Replace non-standard residues (default: True)
            find_missing_residues (bool): Find and handle missing residues (default: True)
            **kwargs: Additional preparation options
            
        Returns:
            Protein: Self for method chaining
            
        Raises:
            ImportError: If PDBFixer is not installed
        """
        if not HAS_PDBFIXER:
            raise ImportError(
                "PDBFixer is required for protein preparation but is not installed. "
                "Install it with: pip install 'pmarlo[fixer]'"
            )
        # Find and replace non-standard residues
        if replace_nonstandard_residues:
            self.fixer.findNonstandardResidues()
            self.fixer.replaceNonstandardResidues()
        
        # Remove heterogens (non-protein molecules)
        if remove_heterogens:
            self.fixer.removeHeterogens(keepWater=keep_water)
        
        # Find and handle missing residues
        if find_missing_residues:
            self.fixer.findMissingResidues()
        
        # Add missing atoms
        if add_missing_atoms:
            self.fixer.findMissingAtoms()
            self.fixer.addMissingAtoms()
        
        # Add missing hydrogens with specified pH
        if add_missing_hydrogens:
            self.fixer.addMissingHydrogens(ph)
        
        self.prepared = True
        
        # Load protein data and calculate properties
        self._load_protein_data()
        self._calculate_properties()
        
        return self
    
    def _load_protein_data(self):
        """Load protein data from the prepared structure."""
        if not self.prepared:
            raise RuntimeError("Protein must be prepared before loading data.")
        
        self.topology = self.fixer.topology
        self.positions = self.fixer.positions
    
    def _calculate_properties(self):
        """Calculate protein properties using RDKit."""
        if self.topology is None:
            return
        
        # Basic topology properties
        self.properties['num_atoms'] = len(list(self.topology.atoms()))
        self.properties['num_residues'] = len(list(self.topology.residues()))
        self.properties['num_chains'] = len(list(self.topology.chains()))
        
        self._calculate_rdkit_properties()
    
    def _calculate_rdkit_properties(self):
        """Calculate properties using RDKit for accurate molecular analysis."""
        try:
            # Use helper function for temporary file handling
            tmp_pdb = self._create_temp_pdb()
            self.rdkit_mol = Chem.MolFromPDBFile(tmp_pdb)
            
            if self.rdkit_mol is not None:
                self._compute_rdkit_descriptors()
            else:
                print("Warning: Could not load molecule into RDKit.")
                
        except Exception as e:
            print(f"Warning: RDKit calculation failed: {e}")
        finally:
            # Clean up temporary file
            if 'tmp_pdb' in locals():
                self._cleanup_temp_file(tmp_pdb)
    
    def _create_temp_pdb(self) -> str:
        """Create a temporary PDB file for RDKit processing."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_file:
            tmp_pdb = tmp_file.name
        
        self.save_prepared_pdb(tmp_pdb)
        return tmp_pdb
    
    def _cleanup_temp_file(self, tmp_file: str):
        """Clean up temporary file."""
        try:
            os.unlink(tmp_file)
        except:
            pass
    
    def _compute_rdkit_descriptors(self):
        """Compute RDKit molecular descriptors."""
        # Calculate exact molecular weight
        self.properties['exact_molecular_weight'] = CalcExactMolWt(self.rdkit_mol)
        
        # Calculate various molecular descriptors
        self.properties['logp'] = Descriptors.MolLogP(self.rdkit_mol)
        self.properties['hbd'] = Descriptors.NumHDonors(self.rdkit_mol)
        self.properties['hba'] = Descriptors.NumHAcceptors(self.rdkit_mol)
        self.properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(self.rdkit_mol)
        self.properties['aromatic_rings'] = Descriptors.NumAromaticRings(self.rdkit_mol)
        self.properties['heavy_atoms'] = Descriptors.HeavyAtomCount(self.rdkit_mol)
        
        # Calculate formal charge
        self.properties['charge'] = Chem.GetFormalCharge(self.rdkit_mol)
        
        # Use exact molecular weight
        self.properties['molecular_weight'] = self.properties['exact_molecular_weight']
    
    def get_rdkit_molecule(self):
        """
        Get the RDKit molecule object if available.
        
        Returns:
            RDKit Mol object or None if not available
        """
        return self.rdkit_mol
    
    def get_properties(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Get protein properties.
        
        Args:
            detailed (bool): Include detailed RDKit descriptors if True
            
        Returns:
            Dict[str, Any]: Dictionary containing protein properties
        """
        properties = self.properties.copy()
        
        if detailed and self.rdkit_mol is not None:
            try:
                properties.update({
                    'tpsa': Descriptors.TPSA(self.rdkit_mol),  # Topological polar surface area
                    'molar_refractivity': Descriptors.MolMR(self.rdkit_mol),
                    'fraction_csp3': Descriptors.FractionCsp3(self.rdkit_mol),
                    'ring_count': Descriptors.RingCount(self.rdkit_mol),
                    'spiro_atoms': Descriptors.NumSpiroAtoms(self.rdkit_mol),
                    'bridgehead_atoms': Descriptors.NumBridgeheadAtoms(self.rdkit_mol),
                    'heteroatoms': Descriptors.NumHeteroatoms(self.rdkit_mol),
                })
            except Exception as e:
                print(f"Warning: Some RDKit descriptors failed: {e}")
        
        return properties
    
    def save(self, output_file: str) -> None:
        """
        Save the protein structure to a PDB file.
        
        Args:
            output_file (str): Path for the output PDB file
        """
        if not self.prepared:
            raise RuntimeError("Protein must be prepared before saving.")
        
        self.save_prepared_pdb(output_file)
    
    def save_prepared_pdb(self, output_file: str) -> None:
        """
        Save the prepared protein structure to a PDB file.
        
        Args:
            output_file (str): Path for the output PDB file
            
        Raises:
            ImportError: If PDBFixer is not installed
            RuntimeError: If protein is not prepared
        """
        if not self.prepared:
            raise RuntimeError("Protein must be prepared before saving. Call prepare() first.")
        
        if not HAS_PDBFIXER:
            raise ImportError(
                "PDBFixer is required for saving prepared structures but is not installed. "
                "Install it with: pip install 'pmarlo[fixer]'"
            )
        
        PDBFile.writeFile(self.fixer.topology, self.fixer.positions, open(output_file, "w"))
    
    def create_system(self, forcefield_files: Optional[list] = None) -> None:
        """
        Create an OpenMM system for the protein.
        
        Args:
            forcefield_files (Optional[list]): List of forcefield files to use
        """
        if not self.prepared:
            raise RuntimeError("Protein must be prepared before creating system.")
        
        if forcefield_files is None:
            forcefield_files = ["amber14-all.xml", "amber14/tip3pfb.xml"]
        
        self.forcefield = ForceField(*forcefield_files)
        self.system = self.forcefield.createSystem(
            self.topology,
            nonbondedMethod=PME,
            constraints=HBonds
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the created system.
        
        Returns:
            Dict[str, Any]: System information
        """
        if self.system is None:
            return {'system_created': False}
        
        forces = {}
        for i, force in enumerate(self.system.getForces()):
            force_name = force.__class__.__name__
            if force_name not in forces:
                forces[force_name] = 0
            forces[force_name] += 1
        
        return {
            'system_created': True,
            'num_forces': self.system.getNumForces(),
            'forces': forces,
            'num_particles': self.system.getNumParticles()
        }
