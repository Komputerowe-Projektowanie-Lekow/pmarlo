### Added
- Shared sampler package at pmarlo.samplers exposing BalancedTempSampler for both feature and trainer layers.
- Central pair-construction helpers under pmarlo.pairs.core with unit coverage ensuring diagnostics and weights logic stay stable.

### Changed
- Legacy DeepTICA facade now delegates pair building, training, and dataset wiring to the core modules, trimming bespoke helpers in _full.py.
- `train_deeptica` now routes training through `DeepTICACurriculumTrainer`, replacing the legacy Lightning fallback while keeping telemetry and whitening metadata.
- pmarlo.features.deeptica_trainer re-exports the canonical ml trainer, with lightweight wrappers plus iter_pair_batches for pairwise batching.
- Feature and shard samplers now hydrate through the shared implementation; PairBuilder.make_pairs resolves indices via build_pair_info for consistent diagnostics.

### Deprecated
- Importing pmarlo.features.samplers or the old trainer utilities now emits deprecation warnings guiding callers to the consolidated modules.
