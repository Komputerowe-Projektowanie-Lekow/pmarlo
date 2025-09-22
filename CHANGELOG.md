
<a id='changelog-0.0.69'></a>
# 0.0.69 — 2025-09-22

REMD: wire end-to-end seeding; app auto-seed per shard; record in provenance.

- API `run_replica_exchange` accepts `random_seed`/`random_state` and forwards to `RemdConfig`.
- App: Simulation Seed mode (fixed | auto | none). Auto generates a unique 32â€‘bit seed per run and logs it.
- Run directories now include the seed (e.g., `run-YYYYMMDD-HHMMSS-seed123`).
- Shard provenance `source` includes `sim_seed` and `seed_mode`.
- Added unit/integration tests for seed propagation and determinism.

Also:
- Add robust resume/chaining: optional checkpoint/PDB restarts, jittered restarts, and safer checkpoint frequency.
- Add diversified starting conditions in the app: Initial PDB, Last frame of run, Random highâ€‘T frame; optional velocity reseed.
- Replica-exchange diagnostics (ladder suggestion, acceptance, diffusion); UI controls for exchange frequency; diagnostics panel with sparkline.
- REMD: honor explicit temperature vectors; validate increasing >0; persist ladder in provenance.json and temps.txt; record schedule mode.
- App: temperature schedule selector (auto-linear, auto-geometric, custom) with Apply toggle; ladder preview and validation; applied to run config.
- Utility: stable geometric ladder generator with tests.

## Added

- created possibility with batch algorithm testing with kubernetes docker desktop kubeadm administrator kernel
- done k8s for the kubernetes with local server

- Optional solvation step that adds an explicit water box when none is present.

## Changed

- made another file for the kubernetes suite with experiment possibility
- changed docker file
- moved md files to the separated directory

- Water molecules are now preserved during protein preparation by default.
<a id='changelog-0.0.36'></a>
# 0.0.36 — 2025-08-30

## Added

- Unified progress callback/reporting (`pmarlo.progress.ProgressReporter`) with ETA and rate limiting.
- Callback kwarg aliases normalized via `coerce_progress_callback`.
- Transform plan serialization helpers: `to_json`, `from_json`, `to_text`.
- Aggregate/build progress events from `pmarlo.transform.runner.apply_plan`.
- Example usage in `example_programs/all_capabilities_demo.py` printing progress.
- Tests for progress reporting, plan serialization, and transform runner events.

## Changed

- `api.run_replica_exchange` accepts `**kwargs` and passes `progress_callback` to the simulation.
- `ReplicaExchange.run_simulation` emits stage events (`setup`, `equilibrate`, `simulate`, `exchange`, `finished`).
- `transform.build.build_result` optionally accepts `progress_callback` to surface aggregate events during transforms.

<a id='changelog-0.14.0'></a>
# 0.14.0 — 2025-08-08

## Added

- psutils for the memory management.

## Changed

- changes in the pyproject.toml and experiments.
- KPIs for the methods and algorithm testing suite upgrades.
- docker now has a lock generations and not just distribution usage.
- made deduplication effort in the probability calculation and logging info from all the modules

<a id='changelog-0.13.0'></a>
# 0.13.0 — 2025-08-08

## Added

- Whole suite for the experimenting with the algoritms(simulation, replica exchange, markov state model) in the docker containers to make them separately run.

<a id='changelog-0.12.0'></a>
# 0.12.0 — 2025-08-08

## Added

- Added the **\[tool.scriv]** section to `pyproject.toml`, setting the format to `md`, the output file to `CHANGELOG.md`, and the fragments directory to `changelog.d`.
