## Fixed
- Deduplicated repeated array concatenation fallback logic by introducing the shared ``concatenate_or_empty`` utility and reusing it within the API and pair-building helpers.
