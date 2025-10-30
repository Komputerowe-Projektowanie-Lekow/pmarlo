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
- The protein should be done before the workflow. When it's not done the workflow cannot proceed.
- When the protein PDBFixer or other modules doen't work the error is raised.
- The protein preparation has all variables explicit not wierd like 7ph things done

# Small explanation
This module is the Protein prep step at the very front of your pipeline (before Transform Plan, REMD, DEMUX,  Shards , …). It prepares a PDB/CIF, validates coordinates, optionally solvates, and exposes quick analytical properties; it also lets you emit a prepared PDB and build an OpenMM System.

init.py exposes a single public class: Protein. The module docstring: “Handles protein preparation, cleanup, and property analysis.”

Tries to import PDBFixer; if unavailable the module raises an ImportError immediately so the workflow stops rather than falling back to a degraded implementation. HAS_NATIVE_PDBFIXER documents whether the import succeeded.

Uses OpenMM (unit, PDBFile, Modeller, ForceField, PME, HBonds) and RDKit for descriptors when requested.

# Key values

# Best practices

# Wanted outcomes

# Examples

# Files description


# Maintenance and evolution
