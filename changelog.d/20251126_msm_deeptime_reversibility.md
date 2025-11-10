## changed
- Align the deeptime-backed MSM estimation so both `MarkovStateModel` and `build_simple_msm`
  use `MaximumLikelihoodMSM(reversible=False)`, matching the reference estimator used in the
  unit test and avoiding the earlier reversible regularization that skewed the transition
  matrix and stationary distribution comparison.

## fixed
- Compute the detailed-balance MAD inside `experiments.kpi` with numpy instead of relying on
  `sklearn.metrics.mean_absolute_error`, so the KPI benchmark stays reproducible even when the
  heavier dependency cannot be loaded and the experiments KPI test keeps its checkpoint valid.
