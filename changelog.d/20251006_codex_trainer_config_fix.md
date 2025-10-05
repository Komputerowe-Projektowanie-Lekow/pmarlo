## Fixed
- Avoided post-init mutation in TrainerConfig so frozen CurriculumConfig instances accept normalised fields and keep caller-defined tau order without raising.

## Changed
- Promoted mlcolvar and scikit-learn to default dependencies and pruned extras that previously re-declared them.
