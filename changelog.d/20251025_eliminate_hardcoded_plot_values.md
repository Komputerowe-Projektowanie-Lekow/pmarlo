## Improved

- Eliminated all hardcoded "magic numbers" from `pmarlo.reporting.plots` module by adding comprehensive plotting constants to `pmarlo.constants`. All figure sizes, DPI values, line widths, contour levels, FES thresholds, and binning parameters are now centralized and configurable via the constants module, improving maintainability and making it easier to tune visualization defaults across the entire codebase.

