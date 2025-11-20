### removed
- Dropped the redundant `pmarlo.markov_state_model._msm_utils` proxy file; MSM helpers now live in `pmarlo.utils.msm_utils` and are imported directly.

### changed
- Updated MSM consumers (API, webapp backend, transforms, tests, and CK utilities) to reference the shared `pmarlo.utils.msm_utils` module.
