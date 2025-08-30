# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

# PDBFixer is optional - users can install with: pip install "pmarlo[fixer]"
try:
    from pdbfixer import PDBFixer

    HAS_PDBFIXER = True
except ImportError:
    PDBFixer = None
    HAS_PDBFIXER = False
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Fixed: Added missing imports for PME and HBonds
from openmm import unit
from openmm.app import PME, ForceField, HBonds, PDBFile
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt


class Protein:
    def __init__(
        self,
        pdb_file: str,
        ph: float = 7.0,
        auto_prepare: bool = True,
        preparation_options: Optional[Dict[str, Any]] = None,
        random_state: int | None = None,
    ):
        """Initialize a Protein object with a PDB file.

        Args:
            pdb_file: Path to the PDB file
            ph: pH value for protonation state (default: 7.0)
            auto_prepare: Automatically prepare the protein (default: True)
            preparation_options: Custom preparation options
            random_state: Included for API compatibility; currently unused.

        Raises:
            ValueError: If the PDB file does not exist, is empty, or has an invalid
                extension
        """
        # If automatic preparation is requested but PDBFixer isn't available,
        # fail fast with a clear ImportError (test expectation when fixer missing).
        if auto_prepare and not HAS_PDBFIXER:
            raise ImportError(
                (
                    "PDBFixer is required for protein preparation but is not "
                    "installed. Install it with: pip install 'pmarlo[fixer]' or "
                    "set auto_prepare=False to skip preparation."
                )
            )

        pdb_path = self._resolve_pdb_path(pdb_file)
        self.random_state = random_state
        self._validate_file_exists(pdb_path)
        self._validate_extension(pdb_path)
        self._validate_readable_nonempty(pdb_path)
        self._validate_ph(ph)
        self._assign_basic_fields(pdb_path, ph)
        self._initialize_fixer(auto_prepare, pdb_file)
        self._initialize_storage()
        self._initialize_properties_dict()
        self._maybe_prepare(auto_prepare, preparation_options, ph)
        if not auto_prepare:
            self._load_basic_properties_without_preparation()

    # --- Initialization helpers to reduce complexity ---

    def _resolve_pdb_path(self, pdb_file: str) -> Path:
        return Path(pdb_file)

    def _validate_file_exists(self, pdb_path: Path) -> None:
        if not pdb_path.exists():
            raise ValueError(f"Invalid PDB path: {pdb_path}")

    def _validate_extension(self, pdb_path: Path) -> None:
        if pdb_path.suffix.lower() not in {".pdb", ".cif"}:
            raise ValueError(f"Unsupported protein file type: {pdb_path.suffix}")

    def _validate_readable_nonempty(self, pdb_path: Path) -> None:
        try:
            with open(pdb_path, "rb") as fh:
                head = fh.read(64)
                if not head.strip():
                    raise ValueError("Protein file is empty")
        except OSError as exc:
            raise ValueError(f"Cannot read protein file: {pdb_path}") from exc

    def _validate_ph(self, ph: float) -> None:
        if not (0.0 <= ph <= 14.0):
            raise ValueError(f"pH must be between 0 and 14, got {ph}")

    def _validate_coordinates(self, positions) -> None:
        if positions is None:
            raise ValueError("No coordinates provided")
        for i, pos in enumerate(positions):
            if pos is None:
                raise ValueError(f"Atom {i} has undefined coordinates")
            try:
                coords = pos.value_in_unit(unit.nanometer)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid coordinate for atom {i}: {exc}") from exc
            if any(
                math.isnan(c) or math.isinf(c) for c in (coords.x, coords.y, coords.z)
            ):
                raise ValueError(f"Atom {i} has non-finite coordinates")

    def _assign_basic_fields(self, pdb_path: Path, ph: float) -> None:
        self.pdb_file = str(pdb_path)
        self.ph = ph

    def _initialize_fixer(self, auto_prepare: bool, pdb_file: str) -> None:
        if not HAS_PDBFIXER:
            self._configure_state_no_fixer(auto_prepare)
        else:
            self._initialize_fixer_instance(pdb_file)

    def _configure_state_no_fixer(self, auto_prepare: bool) -> None:
        # When not preparing automatically, ensure the file exists so
        # invalid paths error early.
        if not auto_prepare and not os.path.isfile(self.pdb_file):
            raise ValueError(f"Invalid PDB path: {self.pdb_file}")
        self.fixer = None
        self.prepared = False
        if auto_prepare:
            raise ImportError(
                (
                    "PDBFixer is required for protein preparation but is not "
                    "installed. Install it with: pip install 'pmarlo[fixer]' "
                    "or set auto_prepare=False to skip preparation."
                )
            )

    def _initialize_fixer_instance(self, pdb_file: str) -> None:
        # PDBFixer will validate the file path and raise appropriately if invalid.
        self.fixer = PDBFixer(filename=pdb_file)
        self.prepared = False

    def _initialize_storage(self) -> None:
        # Store protein data
        self.topology = None
        self.positions = None
        self.forcefield = None
        self.system = None
        # RDKit molecule object for property calculations
        self.rdkit_mol = None
        # Cache for RDKit-derived descriptors
        self._rdkit_properties: Dict[str, Any] = {}

    def _initialize_properties_dict(self) -> None:
        # Protein properties
        self.properties = {
            "num_atoms": 0,
            "num_residues": 0,
            "num_chains": 0,
            "molecular_weight": 0.0,
            "exact_molecular_weight": 0.0,
            "charge": 0.0,
            "isoelectric_point": 0.0,
            "hydrophobic_fraction": 0.0,
            "aromatic_residues": 0,
            "heavy_atoms": 0,
        }

    def _maybe_prepare(
        self,
        auto_prepare: bool,
        preparation_options: Optional[Dict[str, Any]],
        ph: float,
    ) -> None:
        if auto_prepare:
            prep_options = preparation_options or {}
            prep_options.setdefault("ph", ph)
            self.prepare(**prep_options)

    def _load_basic_properties_without_preparation(self) -> None:
        """Load topology with MDTraj and compute basic properties when not prepared.

        This mirrors the previous inline initialization logic for the
        auto_prepare=False path without increasing __init__ complexity.
        """
        try:
            import mdtraj as md
        except Exception as e:
            print(f"Warning: MDTraj not available: {e}")
            return

        try:
            traj = md.load(self.pdb_file)
        except Exception as e:  # pragma: no cover - error path
            raise ValueError(f"Failed to parse PDB file: {e}") from e

        import numpy as np

        if not np.isfinite(traj.xyz).all():
            raise ValueError("PDB contains invalid (non-finite) coordinates")

        topo = traj.topology
        self.properties["num_atoms"] = traj.n_atoms
        self.properties["num_residues"] = topo.n_residues
        # MDTraj topology.chains is an iterator; use n_chains for count
        self.properties["num_chains"] = topo.n_chains

        # Compute approximate molecular weight (sum of atomic masses) and heavy atom count
        total_mass = 0.0
        heavy_atoms = 0
        for atom in topo.atoms:
            # Some elements may have mass None; treat as 0.0
            mass = getattr(atom.element, "mass", None)
            if mass is None:
                mass = 0.0
            total_mass += float(mass)
            if getattr(atom.element, "number", 0) != 1:
                heavy_atoms += 1
        self.properties["molecular_weight"] = total_mass
        self.properties["exact_molecular_weight"] = total_mass
        self.properties["heavy_atoms"] = heavy_atoms

        sequence = self._sequence_from_topology(topo)
        metrics = self._compute_protein_metrics(sequence)
        self.properties.update(metrics)

    def prepare(
        self,
        ph: float = 7.0,
        remove_heterogens: bool = True,
        keep_water: bool = True,
        add_missing_atoms: bool = True,
        add_missing_hydrogens: bool = True,
        replace_nonstandard_residues: bool = True,
        find_missing_residues: bool = True,
        solvate: bool = False,
        solvent_padding: float = 1.0,
        **kwargs,
    ) -> "Protein":
        """
        Prepare the protein structure with specified options.

        Args:
            ph (float): pH value for protonation state (default: 7.0)
            remove_heterogens (bool): Remove non-protein molecules (default: True)
            keep_water (bool): Keep water molecules if True (default: True)
            add_missing_atoms (bool): Add missing atoms to residues (default: True)
            add_missing_hydrogens (bool): Add missing hydrogens (default: True)
            replace_nonstandard_residues (bool): Replace non-standard residues
                (default: True)
            find_missing_residues (bool): Find and handle missing residues
                (default: True)
            solvate (bool): Add an explicit water box if no waters are present
                (default: False)
            solvent_padding (float): Padding in nanometers for the solvent box
                when solvation is requested (default: 1.0)
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

        # Fixed: Added type check to ensure fixer is not None before using it
        if self.fixer is None:
            raise RuntimeError("PDBFixer object is not initialized")

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

        # Optionally solvate the system if no waters are present
        if solvate:
            water_residues = {"HOH", "H2O", "WAT"}
            has_water = any(
                res.name in water_residues for res in self.fixer.topology.residues()
            )
            if not has_water:
                self.fixer.addSolvent(padding=solvent_padding * unit.nanometer)

        self.prepared = True

        # Load protein data and calculate properties
        self._load_protein_data()
        self._calculate_properties()

        return self

    def _load_protein_data(self):
        """Load protein data from the prepared structure."""
        if not self.prepared:
            raise RuntimeError("Protein must be prepared before loading data.")

        # Fixed: Ensure fixer is not None before using it
        if self.fixer is None:
            raise RuntimeError("PDBFixer object is not initialized")

        self.topology = self.fixer.topology
        self.positions = self.fixer.positions
        self._validate_coordinates(self.positions)

    def _calculate_properties(self):
        """Calculate basic protein properties."""
        if self.topology is None:
            return

        self.properties["num_atoms"] = len(list(self.topology.atoms()))
        self.properties["num_residues"] = len(list(self.topology.residues()))
        self.properties["num_chains"] = len(list(self.topology.chains()))

        total_mass = 0.0
        heavy_atoms = 0
        for atom in self.topology.atoms():
            mass = getattr(atom.element, "mass", None)
            if mass is None:
                mval = 0.0
            else:
                try:
                    # OpenMM uses unit-bearing quantities for atomic masses
                    mval = float(mass.value_in_unit(unit.dalton))  # type: ignore[attr-defined]
                except Exception:
                    mval = float(mass)
            total_mass += mval
            if getattr(atom.element, "number", 0) != 1:
                heavy_atoms += 1
        self.properties["molecular_weight"] = total_mass
        self.properties["exact_molecular_weight"] = total_mass
        self.properties["heavy_atoms"] = heavy_atoms

        sequence = self._sequence_from_topology(self.topology)
        metrics = self._compute_protein_metrics(sequence)
        self.properties.update(metrics)

    def _calculate_rdkit_properties(self) -> Dict[str, Any]:
        """Calculate properties using RDKit for accurate molecular analysis."""
        props: Dict[str, Any] = {}
        try:
            tmp_pdb = self._create_temp_pdb()
            self.rdkit_mol = Chem.MolFromPDBFile(tmp_pdb)

            if self.rdkit_mol is not None:
                props = self._compute_rdkit_descriptors()
            else:
                print("Warning: Could not load molecule into RDKit.")

        except Exception as e:
            print(f"Warning: RDKit calculation failed: {e}")
        finally:
            if "tmp_pdb" in locals():
                self._cleanup_temp_file(tmp_pdb)

        self._rdkit_properties = props
        return props

    def _create_temp_pdb(self) -> str:
        """Create a temporary PDB file for RDKit processing."""
        import shutil
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_file:
            tmp_pdb = tmp_file.name

        if self.prepared and HAS_PDBFIXER and self.fixer is not None:
            self.save_prepared_pdb(tmp_pdb)
        else:
            shutil.copy2(self.pdb_file, tmp_pdb)
        return tmp_pdb

    def _cleanup_temp_file(self, tmp_file: str):
        """Clean up temporary file."""
        try:
            os.unlink(tmp_file)
        except Exception:
            pass

    # --- Protein-specific descriptor helpers ---

    def _sequence_from_topology(self, topology) -> str:
        """Extract amino acid sequence from a topology object."""
        aa3_to1 = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLU": "E",
            "GLN": "Q",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
        }

        residues_iter = getattr(topology, "residues", [])
        if callable(residues_iter):
            residues_iter = residues_iter()

        sequence = []
        for res in residues_iter:
            code = aa3_to1.get(res.name.upper(), "")
            if code:
                sequence.append(code)
        return "".join(sequence)

    def _compute_protein_metrics(self, sequence: str) -> Dict[str, Any]:
        """Compute protein-specific metrics from an amino acid sequence."""
        if not sequence:
            return {
                "charge": 0.0,
                "isoelectric_point": 0.0,
                "hydrophobic_fraction": 0.0,
                "aromatic_residues": 0,
            }

        counts = {aa: sequence.count(aa) for aa in set(sequence)}
        n_res = len(sequence)

        hydrophobic = set("AVILMFYWPG")
        aromatic = set("FYW")

        num_hydrophobic = sum(counts.get(aa, 0) for aa in hydrophobic)
        num_aromatic = sum(counts.get(aa, 0) for aa in aromatic)

        hydrophobic_fraction = num_hydrophobic / n_res if n_res else 0.0

        # pKa values for side chains, N-terminus, C-terminus
        pka_side = {
            "C": 8.3,
            "D": 3.9,
            "E": 4.1,
            "H": 6.0,
            "K": 10.5,
            "R": 12.5,
            "Y": 10.1,
        }
        pka_n = 9.69
        pka_c = 2.34

        def charge_at_ph(ph: float) -> float:
            pos = 10 ** (pka_n - ph) / (1 + 10 ** (pka_n - ph))
            neg = 10 ** (ph - pka_c) / (1 + 10 ** (ph - pka_c))
            for aa, count in counts.items():
                if aa in ["K", "R", "H"]:
                    pk = pka_side[aa]
                    pos += count * (10 ** (pk - ph) / (1 + 10 ** (pk - ph)))
                elif aa in ["D", "E", "C", "Y"]:
                    pk = pka_side[aa]
                    neg += count * (10 ** (ph - pk) / (1 + 10 ** (ph - pk)))
            return pos - neg

        # Estimate pI by scanning pH 0-14
        pI = 0.0
        min_charge = float("inf")
        for pH in [x / 100 for x in range(0, 1401)]:
            c = abs(charge_at_ph(pH))
            if c < min_charge:
                min_charge = c
                pI = pH

        charge = charge_at_ph(self.ph)

        return {
            "charge": charge,
            "isoelectric_point": pI,
            "hydrophobic_fraction": hydrophobic_fraction,
            "aromatic_residues": num_aromatic,
        }

    def _compute_rdkit_descriptors(self):
        """Compute RDKit molecular descriptors."""
        props: Dict[str, Any] = {}
        props["exact_molecular_weight"] = CalcExactMolWt(self.rdkit_mol)
        props["molecular_weight"] = props["exact_molecular_weight"]
        props["logp"] = Descriptors.MolLogP(self.rdkit_mol)
        props["hbd"] = Descriptors.NumHDonors(self.rdkit_mol)
        props["hba"] = Descriptors.NumHAcceptors(self.rdkit_mol)
        props["rotatable_bonds"] = Descriptors.NumRotatableBonds(self.rdkit_mol)
        props["aromatic_rings"] = Descriptors.NumAromaticRings(self.rdkit_mol)
        props["heavy_atoms"] = Descriptors.HeavyAtomCount(self.rdkit_mol)
        props["charge"] = Chem.GetFormalCharge(self.rdkit_mol)
        return props

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

        if detailed:
            if not self._rdkit_properties:
                self._rdkit_properties = self._calculate_rdkit_properties()
            properties.update(self._rdkit_properties)

            if self.rdkit_mol is not None:
                try:
                    properties.update(
                        {
                            "tpsa": Descriptors.TPSA(self.rdkit_mol),
                            "molar_refractivity": Descriptors.MolMR(self.rdkit_mol),
                            "fraction_csp3": Descriptors.FractionCsp3(self.rdkit_mol),
                            "ring_count": Descriptors.RingCount(self.rdkit_mol),
                            "spiro_atoms": Descriptors.NumSpiroAtoms(self.rdkit_mol),
                            "bridgehead_atoms": Descriptors.NumBridgeheadAtoms(
                                self.rdkit_mol
                            ),
                            "heteroatoms": Descriptors.NumHeteroatoms(self.rdkit_mol),
                        }
                    )
                except Exception as e:
                    print(f"Warning: Some RDKit descriptors failed: {e}")

        return properties

    def save(self, output_file: str) -> None:
        """
        Save the protein structure to a PDB file.

        If the protein has been prepared with PDBFixer, saves the prepared structure.
        Otherwise, copies the original input file.

        Args:
            output_file (str): Path for the output PDB file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.prepared:
            # If not prepared, copy the original file
            import shutil

            shutil.copy2(self.pdb_file, output_path)
            return

        # For prepared structures, use PDBFixer
        if not HAS_PDBFIXER:
            raise ImportError(
                (
                    "PDBFixer is required for saving prepared structures but is "
                    "not installed. Install it with: pip install 'pmarlo[fixer]'"
                )
            )

        if self.fixer is None:
            raise RuntimeError("PDBFixer object is not initialized")

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
            raise RuntimeError(
                "Protein must be prepared before saving. Call prepare() first."
            )

        if not HAS_PDBFIXER:
            raise ImportError(
                "PDBFixer is required for saving prepared structures but is not installed. "
                "Install it with: pip install 'pmarlo[fixer]'"
            )

        # Fixed: Ensure fixer is not None before using it
        if self.fixer is None:
            raise RuntimeError("PDBFixer object is not initialized")
        self._validate_coordinates(self.fixer.positions)

        with open(output_file, "w") as handle:
            PDBFile.writeFile(self.fixer.topology, self.fixer.positions, handle)

    def create_system(self, forcefield_files: Optional[list] = None) -> None:
        """
        Create an OpenMM system for the protein.

        If the protein has not been prepared, loads topology directly from the
        input PDB file. Otherwise, uses the prepared topology.

        Args:
            forcefield_files (Optional[list]): List of forcefield files to use
        """
        try:
            # Load topology if not already loaded
            if self.topology is None:
                # Load topology and positions directly from the input PDB file
                pdb = PDBFile(self.pdb_file)
                self.topology = pdb.topology
                self.positions = pdb.positions
                self._validate_coordinates(self.positions)

            if forcefield_files is None:
                forcefield_files = ["amber14-all.xml", "amber14/tip3pfb.xml"]

            self.forcefield = ForceField(*forcefield_files)
            if self.forcefield is None:
                raise RuntimeError("ForceField could not be created")

            self.system = self.forcefield.createSystem(
                self.topology, nonbondedMethod=PME, constraints=HBonds
            )
        except Exception as e:
            print(f"Warning: System creation failed: {e}")
            self.system = None

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the created system.

        Returns:
            Dict[str, Any]: System information
        """
        if self.system is None:
            return {"system_created": False}

        forces = {}
        for i, force in enumerate(self.system.getForces()):
            force_name = force.__class__.__name__
            if force_name not in forces:
                forces[force_name] = 0
            forces[force_name] += 1

        return {
            "system_created": True,
            "num_forces": self.system.getNumForces(),
            "forces": forces,
            "num_particles": self.system.getNumParticles(),
        }
