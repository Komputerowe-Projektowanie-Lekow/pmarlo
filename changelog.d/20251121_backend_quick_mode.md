## fixed

- Ensure quick-mode sampling obtains `run_replica_exchange` from the backend module at runtime so the default quick workflow still executes a real REMD run when requested and unit tests that patch `pmarlo_webapp.app.backend.run_replica_exchange` now observe the mocked engine (`pmarlo_webapp/app/backend/sampling.py`).
