### Changed
- Refactored `find_local_minima_2d` in `pmarlo.markov_state_model.picker` to use SciPy filters for neighborhood comparisons, removing nested Python loops for improved performance and clarity.

### Added
- Unit tests covering edge cases and invalid-value handling for `find_local_minima_2d` in `tests/unit/markov_state_model/test_picker.py`.
