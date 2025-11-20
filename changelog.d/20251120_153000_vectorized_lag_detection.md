### changed
- Vectorized `select_lag_from_its` plateau detection to avoid Python-level loops while preserving existing stability checks.

### added
- Unit coverage for plateau detection edge cases, including stable regions and invalid timescale entries.
