added:
- Enabled `run_single_temperature_md` to export restart snapshots so single-temperature runs in the app can save checkpointable final structures just like REMD jobs.
- Added regression tests covering the single-temperature snapshot flow and the helper that resolves the effective run temperature.
- Added `_run_single_temp_production` method to `ReplicaExchange` class to properly handle production phase when `n_replicas == 1` or `exchange_frequency` is very large.
- Added comprehensive unit tests for single-temperature MD production phase execution.
fixed:
- Streamlit workflow backend now routes single-temperature runs through dedicated logic, preserving the config flag on resume, recording accurate ladder metadata, and preventing spurious "temperature ladder must have two values" failures.
- Fixed production phase not running any MD steps when using single-temperature MD (when `n_replicas == 1` or `exchange_frequency` is larger than production steps). Production duration was 0 ms because `exchange_steps` was 0.
- Fixed `_log_final_stats` to skip inappropriate REMD-specific warnings (exchange acceptance and replica diffusion) when running single-temperature MD.
- Single-temperature MD simulations now properly execute production steps and report completion without spurious warnings.
