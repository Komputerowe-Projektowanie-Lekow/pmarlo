## Added
- Optional REMD restart snapshot export hook that writes the last simulation frame to a PDB for reuse.
- Streamlit app controls to persist final-frame PDBs into the inputs catalog and load them for follow-up runs.

## Fixed
- Ensured stub sampling path mirrors restart snapshot behaviour so quick tests exercise restart wiring.
