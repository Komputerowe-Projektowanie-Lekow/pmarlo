## changed
- Restored the `pmarlo.api` facade to re-export the public helpers now housed in the package modules so existing imports keep working.
- Updated example programs and the Streamlit webapp backend to import helpers from the new package submodules.
- Promoted `choose_sim_seed` into `pmarlo.utils.seed` and exposed it via `pmarlo.api` so every client (including the webapp) relies on the same simulation seeding logic.

## removed
- Dropped the legacy `pmarlo/api.py` god-module in favour of the new modular API package structure.
