### Fixed

- Bound the TICA performance harness to pytest's `tmp_path`, eliminating the
  repository-level `tmp/` artefacts it previously produced during perf test
  runs and ensuring its feature cache lives alongside the invoking test context.
- Audited replica-exchange and MSM perf harnesses to confirm no code path still
  seeds a `tmp_ms/` directory at the project root.
