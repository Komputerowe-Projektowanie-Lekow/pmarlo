## Changed
- Locked tooling envs to Python 3.12 in tox and tightened project requires-python to hold back 3.13 auto-detection.
- Centralised flake8 configuration with targeted per-file ignores to unblock test imports while planning deeper refactors.

## Fixed
- Resolved deeptica trainer type regressions by aliasing optional imports, normalising tuple defaults, and formalising dataset fallbacks for mypy.
- Removed unused typing artifacts and ensured StandardScaler fallbacks conform to the runtime interface expected by feature preparation.
