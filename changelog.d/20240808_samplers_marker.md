### Fixed
- Registered missing pytest markers (`analysis`, `samplers`, `pdbfixer`, `tica`) so the performance suite collects without errors.
- Converted the perf exchange algorithm benchmark banner into a comment to avoid syntax errors under strict collection.
- Added typing support for YAML configuration loading and tightened REMD helper utilities so mypy's ``type`` environment runs cleanly.
