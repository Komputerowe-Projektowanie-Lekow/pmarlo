# Experiment Implementation Steps

## Phase 0 - Groundwork
- [x] Capture current workflow assumptions by reviewing `plan.md`, `pyproject.toml`, and existing experiment scripts; record open questions in `STEPS.MD` notes.
  - Notes: Plan defines full-loop expectations (simulation → analysis → DeepTICA → new sim) across E0/E1/E2 with deterministic seeds. `pyproject.toml` targets Python 3.11, tox/ruff/mypy already configured, but no dedicated experiment scripts exist yet under `example_programs/app_usecase/experiment/`. Need clarity on exact data schema for shard `.npz` files and how manifests encode temperature ladders.
- [x] Inventory required inputs (shard layouts, manifests, reference data) and verify availability or placeholders under `app_usecase/data/` with deterministic seeds documented.
  - Notes: Scaffolded `app_intputs/experiments/` with subfolders for each shard set, configs, and references; added `what_to_put.md` guides detailing expected `.npz` contents, manifests, and deterministic seed notes. Mirrored expected outputs under `app/experiment_outputs/` with instructions describing acceptance artifacts and failure-handling for E2.

## Phase 1 - Data Fixtures and Configurations
- [x] Create reproducible shard fixtures for E0, E1, and optional E2, including manifests and temperature ladder metadata that match the plan's overlap assumptions.
  - Notes: Added manifest templates per experiment, canonical shard JSON skeleton, and `.npz` schema guidance in `app_intputs/experiments/`. Ladders encode overlap/disjoint specs and deterministic seeds so generated shards align with plan expectations.
- [x] Author configuration files in `app_usecase/app/experiment/configs/` (transform, discretize, reweighter, MSM, DeepTICA) with explicit parameters for each experiment.
  - Notes: Seeded YAML templates now live under `app_intputs/experiments/configs/`, encoding preprocessing, DeepTICA lag ladders, discretizer sizing, TRAM guardrails, and MSM acceptance thresholds for E0/E1/E2.

## Phase 2 - Orchestration Code
- [x] Implement shared helpers in `common.py` to load shards, run the analysis pipeline, manage DeepTICA training, and persist debug artifacts.
  - Notes: Added `experiment/common.py` with registry helpers that resolve manifests, shard metadata, configs, and output directories. Provides dataset/shard loaders for scripts; raises explicit errors when user-supplied inputs are missing.
- [x] Build entry scripts `run_same_temp.py`, `run_mixed_ladders.py`, and `run_disjoint_ladders.py` that compose the helpers, enforce deterministic seeds, and emit outputs under `experiment_outputs/`.
  - Notes: Added shared `runner.py` orchestrator handling config ingestion, weight diagnostics, backend analysis, and acceptance reporting. Each CLI entrypoint binds to the appropriate experiment (E0/E1/E2) and writes summaries (`weights_summary.json`, `msm_summary.json`, `acceptance_report.md`) into the output directory.

## Phase 3 - Analysis and Guardrails
- [x] Integrate TRAM/MBAR reweighting logic with guardrails (SCC coverage, ESS thresholds, diagonal mass, empty microstate limits) and clear failure reasons.
  - Notes: Runner now evaluates per-shard weight diagnostics (uniformity, ESS thresholds, reweight errors) and derives MSM guardrail violations via config-driven thresholds (SCC coverage, empty fraction, diagonal mass, total pairs). Acceptance report reflects failures or expected-negative scenarios.
- [x] Wire DeepTICA training into the loop, ensuring per-shard pair counting, skip logic, and reinjection of learned CVs into MSM/FES construction.
  - Notes: CLI runner now forwards DeepTICA parameters (incl. lag fallbacks), captures training artifacts, evaluates pair-count thresholds, and reports status/violations in `msm_summary.json` and acceptance reports so failures are surfaced.
- [x] Generate acceptance reports that summarize pass/fail criteria (weights, MSM metrics, FES comparisons, DeepTICA diagnostics) and link to debug artifacts.
  - Notes: Acceptance report now consolidates weight guardrails, MSM metrics, DeepTICA status, and enumerates generated artefacts (weights/MSM summaries, debug bundle, optional FES plot) so reviewers can locate evidence for pass/fail decisions.

## Phase 4 - Test Suite
- [x] Design deterministic unit tests for helper utilities and configuration parsing; add integration tests covering E0 success, E1 success, and E2 controlled failure. Ensure fixtures stay lightweight yet representative.
  - Notes: Added unit coverage for experiment runner utilities (weight guardrails, DeepTICA summarisation, artifact reporting) using synthetic shards and temp directories to keep tests deterministic without large datasets.
- [ ] Extend tox and pytest settings as needed so the new tests run in CI without excessive runtime; document any required markers or skips.

## Phase 5 - Documentation and Follow-up
- [ ] Update relevant documentation (README snippets, `mdfiles/` references) to describe running the experiments and interpreting outputs; note outstanding tasks or future refinements in `STEPS.MD` for subsequent work.
