### Fixed

- Resolved 7 merge conflicts in `src/pmarlo/protein/protein.py` consolidating strict
  PDBFixer handling. The module now consistently fails fast with clear ImportError
  messages when PDBFixer is required but unavailable, with no silent fallbacks.


