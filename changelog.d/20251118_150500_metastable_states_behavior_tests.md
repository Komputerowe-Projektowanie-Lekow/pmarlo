### fixed
- Hardened `_find_metastable_states` against invalid inputs by validating temperature, state indices, flux coverage, and KIS data, while also labeling overlapping source/sink states as `source_sink`.

### added
- Black-box behavioral tests for `_find_metastable_states` covering metadata propagation, optional KIS/macrostate handling, empty state protection, and error cases for invalid inputs.
