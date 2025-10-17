# Experiment Configuration Files

Base configuration templates have been added for you:

- `transform_plan.yaml`: per-experiment preprocessing and DeepTICA settings (E0/E1/E2).
- `discretize.yaml`: shared clustering defaults plus experiment-specific guardrails.
- `reweighter.yaml`: identity/TRAM modes with ESS tolerances and diagnostics.
- `msm.yaml`: MSM estimation options and acceptance thresholds.

Adjust numeric values as you gather empirical evidence, but keep seeds deterministic so regression tests remain reproducible. If you introduce additional configs, document them here.
