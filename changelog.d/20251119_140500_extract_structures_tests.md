### added
- Added a pytest suite exercising `RepresentativePicker.extract_structures` across in-memory, locator-driven, and failure scenarios.

### fixed
- Updated `RepresentativeFrame` typing and method definition formatting so raw-trajectory representatives with missing local indices are fully supported without mypy/type-check noise.
