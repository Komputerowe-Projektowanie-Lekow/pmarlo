## Fixed

- Updated `_clamp01` in `pmarlo.experiments.kpi` to clamp values with
  `numpy.clip`, relying on the standard NumPy helper instead of manual bounds
  checks.
