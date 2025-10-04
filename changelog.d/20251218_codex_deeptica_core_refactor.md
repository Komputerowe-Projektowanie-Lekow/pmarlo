### Added
- Core DeepTICA trainer package modules (`config`, `loops`, `sampler`, `schedulers`, `trainer`) and focused unit tests under `tests/unit/features/deeptica/core/` covering each helper.
- Lightweight README for `src/pmarlo/features/deeptica/core/` documenting module responsibilities.

### Changed
- Diagnostics now reuse `features.deeptica.core.pairs.build_pair_info` for uniform pair statistics while keeping bias-aware reporting intact.
- Public trainer API continues to live at `pmarlo.features.deeptica_trainer` via the new package re-export.
