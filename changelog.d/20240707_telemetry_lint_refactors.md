### Changed
- Export DeepTICA helper symbols directly from the package namespace and simplify lazy loading bookkeeping.
- Reworked DeepTICA feature canonicalisation to use a dedicated collector, reducing branching complexity.
- Introduced a reusable progress reporter for pipeline stage updates to lower cyclomatic complexity in the handler.
- Refactored REMD orchestration, diagnostics, and shard authoring logic into reusable helpers so `tox -e lint` passes without suppressing cyclomatic thresholds.
