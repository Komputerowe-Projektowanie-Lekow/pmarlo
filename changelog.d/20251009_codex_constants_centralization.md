## Added
- Introduced `pmarlo.constants` as the single source of project-wide physical, numeric, and domain defaults for reuse across subsystems.

## Changed
- Refactored reweighting, MSM, DeeptiCA, replica-exchange, and reporting modules to import shared constants instead of hard-coded literals.
- Standardised epsilon/tolerance guards, display scales, and energy thresholds by referencing the shared constants module.
