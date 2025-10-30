### Changed
- Removed implicit ``output/`` fallbacks across replica exchange, MSM, simulation, and pipeline helpers; callers must now provide explicit output directories (tests and examples updated to persist under ``example_programs/programs_outputs``).
- ``retune_temperature_ladder`` now requires an explicit ``output_json`` path, keeping ladder suggestions alongside the chosen workspace instead of the repository root.
