## Fixed

- Corrected `pmarlo.utils.replica_utils.geometric_ladder` to respect `endpoint=False`,
  matching NumPy's geometric spacing semantics and preventing extraneous
  high-temperature samples in derived ladders.
