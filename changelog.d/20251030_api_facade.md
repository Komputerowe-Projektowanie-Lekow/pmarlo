## changed
- Restored the `pmarlo.api` facade to re-export the public helpers now housed in the package modules so existing imports keep working.
- Updated example programs and the Streamlit webapp backend to import helpers from the new package submodules.

## removed
- Dropped the legacy `pmarlo/api.py` god-module in favour of the new modular API package structure.
