### Fixed

- Addressed multiple mypy failures in the ``type`` tox environment by tightening
  MSM attribute handling, refining clustering estimator typing, and improving
  conformations utilities annotations so ``poetry run tox -e type`` passes again.
