### Fixed
- Resolved merge conflicts in conformations analysis workflow
  - Updated `StateDetector` to use `committor_thresholds` parameter instead of `n_metastable`
  - Maintained backward compatibility by keeping `_resolve_metastable_count` method
  - Updated Streamlit app to support both metastable state configuration and committor thresholds
  - Fixed backend conformations analysis to use new `committor_thresholds` parameter