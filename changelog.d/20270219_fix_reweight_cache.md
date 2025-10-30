## Fixed

- Prevented the analysis reweighter cache from reusing weights when shard thermodynamics or base weights change, avoiding stale
  distributions when reweighting multiple datasets with shared shard identifiers.
