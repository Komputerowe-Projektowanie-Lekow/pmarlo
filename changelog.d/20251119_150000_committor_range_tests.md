### added
- Dedicated behavioral tests for `pick_from_committor_range` that exercise its selection contract, delegation to `pick_representatives`, and all documented error paths.

### fixed
- Normalized `dtrajs` inside `pick_from_committor_range` and tightened dimensionality checks so range validation cannot silently access unsupported inputs before calling the generic picker.
