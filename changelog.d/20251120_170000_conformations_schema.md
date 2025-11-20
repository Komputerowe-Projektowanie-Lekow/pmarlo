### changed
- Conformations summaries are now written through a Pydantic schema, ensuring the saved JSON mirrors the in-memory `ConformationsResult` payload.

### added
- Round-trip coverage for the conformations result schema using the real 3gd8-fixed PDB asset.
