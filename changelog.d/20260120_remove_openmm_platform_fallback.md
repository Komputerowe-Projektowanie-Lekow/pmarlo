### Fixed
- Removed the silent OpenMM platform fallback in `pmarlo.replica_exchange._simulation_full.Simulation` so runs now fail fast when the requested backend is unavailable and only explicit CPU or CUDA selections are accepted.
