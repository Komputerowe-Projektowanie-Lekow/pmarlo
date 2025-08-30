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
- `engine.build.build_result` optionally accepts `progress_callback` to surface aggregate events during transforms.

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
