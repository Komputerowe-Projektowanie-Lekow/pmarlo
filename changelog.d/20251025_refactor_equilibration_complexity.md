# Refactor: Reduce Cyclomatic Complexity in Equilibration Functions

## Changed

- Refactored `_run_gradual_heating` and `_run_temperature_equilibration` methods in `replica_exchange.py` to reduce cyclomatic complexity and improve maintainability
- Extracted single-responsibility helper methods following SOLID principles:
  - `_calculate_heating_parameters`: Calculate heating phase parameters
  - `_log_heating_milestone`: Log heating milestones at threshold percentages
  - `_perform_heating_step`: Execute heating step for all replicas
  - `_report_heating_progress`: Report heating progress to reporter and logger
  - `_complete_heating_phase`: Finalize heating phase with checkpointing
  - `_initialize_temperature_equilibration`: Initialize equilibration parameters
  - `_set_target_temperatures`: Configure replica integrators to target temperatures
  - `_perform_equilibration_step`: Execute equilibration step with error handling
  - `_log_equilibration_milestone`: Log equilibration milestones
  - `_report_equilibration_progress`: Report equilibration progress
  - `_complete_equilibration_phase`: Finalize equilibration with checkpointing

## Improved

- Enhanced code maintainability by separating concerns into focused, testable methods
- Reduced cognitive complexity of equilibration workflow
- Preserved all existing functionality and values without introducing fallbacks
- Improved code readability by extracting complex nested logic into named methods
