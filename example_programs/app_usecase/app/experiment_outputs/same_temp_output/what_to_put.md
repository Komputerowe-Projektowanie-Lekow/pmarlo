# Expected E0 Output Layout

Populate this directory after running `run_same_temp.py`. The pipeline should emit:

- `analysis_debug/summary.json` plus ancillary debug artifacts (state assignments, DeepTICA diagnostics, etc.).
- `fes_Tref.png`: free energy surface at `T_ref` using both baseline and DeepTICA CVs.
- `msm_summary.json`: MSM metrics (timescales, SCC coverage, empty-state fraction).
- `weights_summary.json`: MBAR/TRAM weights (should be near-uniform).
- `acceptance_report.md`: PASS/FAIL checklist documenting all guardrails.

Keep filenames deterministic so regression tests can diff contents reliably.
