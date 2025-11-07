## added
- Introduced `pmarlo.utils.input_parsing` with shared `parse_temperature_ladder` and `parse_tau_schedule` helpers plus regression tests so every client validates user-provided ladders and tau schedules the same way.
- Promoted the shard run selection helper into `pmarlo.data.shard_io.select_shard_paths` and re-exported it via `pmarlo.api` for reuse outside the webapp.

## changed
- Restored the `pmarlo.api` facade to re-export the public helpers now housed in the package modules so existing imports keep working.
- Updated example programs and the Streamlit webapp backend to import helpers from the new package submodules.
- Promoted `choose_sim_seed` into `pmarlo.utils.seed` and exposed it via `pmarlo.api` so every client (including the webapp) relies on the same simulation seeding logic.
- Rewired the Streamlit app to consume the new parsing utilities directly from `pmarlo.api`, eliminating duplicate logic inside `pmarlo_webapp`.
- Updated the Streamlit tabs to consume `pmarlo.api.select_shard_paths`, ensuring shard selections raise actionable errors instead of relying on UI-only helpers.

## removed
- Dropped the legacy `pmarlo/api.py` god-module in favour of the new modular API package structure.
