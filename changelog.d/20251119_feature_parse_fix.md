## fixed

- Align `tests/integration/smoke/test_features_quick.py` expectations with `parse_feature_spec` so both unit and integration suites consume the same `indices` argument semantics.
- Ensure `parse_feature_spec` exposes an `atoms` key for list-style molecular specs so CV parsing consumes consistent metadata within the integration suite.
