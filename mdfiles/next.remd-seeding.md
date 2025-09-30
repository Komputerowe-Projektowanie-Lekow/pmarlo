REMD: wire end-to-end seeding; app auto-seed per shard; record in provenance.

- API `run_replica_exchange` accepts `random_seed`/`random_state` and forwards to `RemdConfig`.
- App: Simulation Seed mode (fixed | auto | none). Auto generates a unique 32‑bit seed per run and logs it.
- Run directories now include the seed (e.g., `run-YYYYMMDD-HHMMSS-seed123`).
- Shard provenance `source` includes `sim_seed` and `seed_mode`.
- Added unit/integration tests for seed propagation and determinism.

Also:
- Add robust resume/chaining: optional checkpoint/PDB restarts, jittered restarts, and safer checkpoint frequency.
- Add diversified starting conditions in the app: Initial PDB, Last frame of run, Random high‑T frame; optional velocity reseed.
- Replica-exchange diagnostics (ladder suggestion, acceptance, diffusion); UI controls for exchange frequency; diagnostics panel with sparkline.
- REMD: honor explicit temperature vectors; validate increasing >0; persist ladder in provenance.json and temps.txt; record schedule mode.
- App: temperature schedule selector (auto-linear, auto-geometric, custom) with Apply toggle; ladder preview and validation; applied to run config.
- Utility: stable geometric ladder generator with tests.
