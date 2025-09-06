REMD: wire end-to-end seeding; app auto-seed per shard; record in provenance.

- API `run_replica_exchange` accepts `random_seed`/`random_state` and forwards to `RemdConfig`.
- App: Simulation Seed mode (fixed | auto | none). Auto generates a unique 32‑bit seed per run and logs it.
- Run directories now include the seed (e.g., `run-YYYYMMDD-HHMMSS-seed123`).
- Shard provenance `source` includes `sim_seed` and `seed_mode`.
- Added unit/integration tests for seed propagation and determinism.

Also:
- Add robust resume/chaining: optional checkpoint/PDB restarts, jittered restarts, and safer checkpoint frequency.
