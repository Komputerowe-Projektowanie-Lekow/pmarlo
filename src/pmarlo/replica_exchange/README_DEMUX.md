PMARLO Demultiplexing (Demux) Overview
=====================================

What Changed
------------
- The demultiplexing logic has been refactored into a streaming, memory‑efficient
  engine with a pure planning layer and pluggable I/O backends.
- A facade keeps the original public API (`ReplicaExchange.demux_trajectories`)
  and routes to the new engine behind a feature flag.
- Metadata has been extended (schema_version=2) to include provenance
  (repairs/skips, fill policy, contiguous real‑data blocks, checksums).

How to Toggle Legacy vs Streaming
---------------------------------
- Streaming is enabled by default.
- Control via module flags in `pmarlo.replica_exchange.config`:
  - `DEMUX_STREAMING_ENABLED: bool = True`
  - `DEMUX_BACKEND: str = "mdtraj"`  (or "mdanalysis" if installed)
  - `DEMUX_FILL_POLICY: str = "repeat"`  ("repeat" | "skip" | "interpolate")
  - `DEMUX_PARALLEL_WORKERS: int | None = None`  (set to 2..N to enable)
  - `DEMUX_CHUNK_SIZE: int = 2048`  (reader chunk and writer rewrite threshold)
- You may also set `remd.demux_io_backend` or `remd.demux_fill_policy` on a
  `ReplicaExchange` instance to override at runtime.
  Additional instance overrides supported:
  - `remd.demux_backend`, `remd.demux_parallel_workers`, `remd.demux_chunk_size`.

Known Limitations
-----------------
- DCD append behavior: A safe append‑like writer is provided. For MDTraj it uses
  a bounded buffer and periodically rewrites the output file (atomic swap).
  For MDAnalysis it writes sequentially to an open handle but is not a true
  in‑place appender. For very large runs, prefer larger rewrite thresholds or a
  backend that supports streaming appends natively.
- Interpolation requires both a previous written frame and the first frame of
  the next segment. When unavailable, the engine falls back to `repeat`.

Adding a New Backend
--------------------
1. Implement `TrajectoryReader` and `TrajectoryWriter` for the backend under
   `pmarlo/io/trajectory_reader.py` and `pmarlo/io/trajectory_writer.py`.
2. Register in the factory helpers:
   - `get_reader(backend: str, topology_path: str|None)`
   - `get_writer(backend: str, topology_path: str|None)`
3. Ensure both raise project‑specific errors with clear messages when optional
   dependencies are missing.
4. Add basic selection tests under `tests/io/`.

Metadata Hints for MSM
----------------------
- `contiguous_blocks` in the metadata v2 enumerates [start, stop) intervals of
  frames with only real (non‑filled) data.
- Use `pmarlo.replica_exchange.demux_hints.load_demux_hints(meta_path)` to load
  these intervals and apply masks in downstream MSM workflows without scanning
  the trajectory.
