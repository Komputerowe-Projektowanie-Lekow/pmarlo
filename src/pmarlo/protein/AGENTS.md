Reporting module:
- unnegotiable
- explanation
- key values
- best practices
- wanted outcomes
- examples
- files description
- maintenance and evolution

# Unnegotiable


# Small explanation
This module is the Protein prep step at the very front of your pipeline (before Transform Plan, REMD, DEMUX,  Shards , …). It prepares a PDB/CIF, validates coordinates, optionally solvates, and exposes quick analytical properties; it also lets you emit a prepared PDB and build an OpenMM System.

init.py exposes a single public class: Protein. The module docstring: “Handles protein preparation, cleanup, and property analysis.”

Tries to import PDBFixer; if unavailable, it installs a minimal stub using OpenMM’s Modeller that can remove heterogens, add hydrogens, and (if forcefields are present) add solvent. Flags HAS_NATIVE_PDBFIXER / USING_PDBFIXER_STUB document which path is active.

Uses OpenMM (unit, PDBFile, Modeller, ForceField, PME, HBonds) and RDKit for descriptors when requested.

# Key values

# Best practices

# Wanted outcomes

# Examples

# Files description


# Maintenance and evolution
