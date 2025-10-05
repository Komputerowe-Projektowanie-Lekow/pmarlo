## Added
- `pmarlo.features.deeptica.core.trainer_api.train_deeptica_pipeline` orchestrates feature prep, pair building, training, and whitening while returning `TrainingArtifacts` for callers.
- Unit test `tests/unit/features/deeptica/core/test_trainer_api.py` exercises the new pipeline with stubbed curriculum trainer dependencies.

## Changed
- `pmarlo.features.deeptica._full.train_deeptica` now delegates to the modular pipeline, reducing duplication and keeping optional dependency handling centralized.
- `pmarlo.features.deeptica.core.dataset.split_sequences` is exported for reuse and the DeepTICA core README documents the trainer API boundaries.
