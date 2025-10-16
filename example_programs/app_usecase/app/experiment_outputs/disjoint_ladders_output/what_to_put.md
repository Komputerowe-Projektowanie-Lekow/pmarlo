# Expected E2 Output Layout (Negative Control)

When `run_disjoint_ladders.py` is executed, this directory should capture the controlled failure case:

- `analysis_debug/summary.json`: records the reweighter failure (e.g., ESS ≈ 0 or explicit exception) and DeepTICA skip reason.
- Any partial debug artifacts that explain the failure (weights arrays, logs) under `analysis_debug/`.
- `acceptance_report.md`: should clearly mark **FAIL** and enumerate which guardrails tripped.

If the pipeline raises before writing some files, create lightweight stubs that document the failure reason so regression tests can assert the correct behaviour.
