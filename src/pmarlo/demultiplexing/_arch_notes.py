"""
Architecture notes for REMD demultiplexing (demux) and trajectory handling.

This module intentionally contains only comments. It serves as lightweight
documentation for the current demultiplexing implementation and its
collaborators to guide future refactors toward a streaming, memory‑efficient,
and resilient design, while preserving backwards compatibility of the public
API.

Current data flow
-----------------
- Entry points:
  - `ReplicaExchange.demux_trajectories(...)` is a thin delegator to
    `pmarlo.demultiplexing.demux.demux_trajectories(remd, ...)`.
  - Downstream analysis code (e.g., MSM loaders) may consume JSON metadata
    written alongside the demultiplexed trajectory via
    `pmarlo.demultiplexing.demux_metadata.DemuxMetadata`.

- Inputs:
  - REMD simulation state object (`ReplicaExchange`) providing:
    - `temperatures`: list[float]
    - `exchange_history`: list[list[int]] — mapping segment -> replica @ temp index
    - `trajectory_files`: list[Path] — per‑replica DCD files
    - `reporter_stride` and `_replica_reporter_stride`: strides relating MD steps
      to saved frames
    - `exchange_frequency`: MD steps between exchanges
    - `integrators`: for retrieving the integration timestep when writing metadata
    - `pdb_file`, `output_dir`

- Processing:
  - Determine closest replica temperature to the requested `target_temperature`.
  - Iterate exchange segments; for each segment, identify which replica was at
    the target temperature and slice frames from that replica's trajectory based
    on stride and the segment's MD step window.
  - Gaps or missing data are handled with simple repair strategies (duplicate
    of the last available frame) and recorded in logs.

- Outputs:
  - A demultiplexed DCD written to `output_dir` with a name encoding the
    temperature, e.g. `demux_T300K.dcd`.
  - A sidecar JSON metadata file `demux_T300K.meta.json` containing
    `DemuxMetadata` with provenance (exchange frequency, timestep, frames per
    segment, temperature schedule).

I/O points
----------
- Reads: per‑replica DCD files via MDTraj (`md.load`), topology via the REMD
  object (`pdb_file`). The loader suppresses plugin chatter via
  `pmarlo.io.trajectory._suppress_plugin_output` when available.
- Writes: demuxed DCD (`Trajectory.save_dcd`) and JSON metadata
  (`Path.write_text` via `DemuxMetadata.to_json`).

Error paths and resilience
--------------------------
- Missing exchange history: returns `None` early with a warning.
- Missing trajectory files: logs warnings. If no trajectories can be loaded,
  returns `None` without raising I/O errors.
- Inconsistent segment times or frame indices: raises
  `DemuxIntegrityError` (project‑specific) to surface logical issues.
- Missing replica at the target temperature or empty segments:
  repairs by duplicating the previous frame when possible; otherwise raises
  `DemuxIntegrityError`.
- Errors while processing a specific segment: emits a `demux_error` progress
  event and continues with subsequent segments (best‑effort demux).
- Errors while saving outputs: logs an error and returns `None`.

Logging and progress reporting
------------------------------
- Uses the package logger (`pmarlo`) at INFO/WARNING/ERROR levels to capture
  high‑level state, diagnostics, and repair actions.
- Emits structured progress events (`demux_begin`, `demux_segment`,
  `demux_gap_fill`, `demux_end`, `demux_error`) via `ProgressReporter` so
  console/UI layers can display ETA and contextual messages consistently.

Performance hot‑spots
---------------------
- Loading full DCDs into memory (`md.load`) and joining many segments can be
  memory‑intensive for large runs.
- Current implementation accumulates segment `Trajectory` objects in memory and
  performs a final `md.join`, which can incur high peak memory.
- I/O write path assumes `save_dcd` of a single, in‑memory `Trajectory`. Appends
  or chunked streaming are not used yet.

Global assumptions
------------------
- Exchange frequency is constant (MD steps between exchange attempts are
  uniform across segments).
- Reporter stride per replica is either globally planned or captured per‑replica
  at setup time; it is used to map MD steps to saved frame indices.
- Equilibration consists of two phases (gradual heating, temperature
  equilibration) whose combined steps form an initial offset before production
  segments begin; demux uses this effective offset when mapping MD step windows
  to frames.

Notes for future refactor
-------------------------
- Introduce streaming demux: iterate over segments and write frames in chunks
  to an output writer that supports appends or a rolling buffer to cap memory.
- Detect and adapt to the capabilities of the chosen trajectory writer (e.g.,
  lack of append support) at runtime; document fallbacks and limitations in
  metadata and logs.
- Tighten invariants and metadata validation (e.g., cross‑checking strides and
  frame counts) and centralize error types.
"""
