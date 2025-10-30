### Fixed
- Honour the requested microstate count when building MSM bundles so the webapp no longer trips the `state_count_mismatch` guardrail during analysis.
- Removed shadowing `import traceback` statements inside the Streamlit app to prevent the UnboundLocalError raised when reporting build failures.
- Regenerated the sanitiser docstring for debug exports to silence the invalid escape sequence warning emitted when launching the webapp.
- Added regression coverage for the microstate plumbing and backend integration to keep the guardrail path intact.
