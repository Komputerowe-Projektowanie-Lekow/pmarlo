### Changed
- Updated `run_conformations_analysis()` in the Streamlit backend to build reversible MSMs with deeptime's maximum likelihood estimator, guaranteeing detailed balance or failing fast when deeptime is unavailable.
- Enforced reversible MSM estimation within shared MSM utilities and kinetic importance MSM rebuilding to eliminate complex eigenvalues in downstream analyses.
