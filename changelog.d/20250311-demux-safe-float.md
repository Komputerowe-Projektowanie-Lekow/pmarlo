- Fixed the demultiplexing engine to skip streaming from source-less segments before attempting I/O and correctly report progress.
- Updated `safe_float` to gracefully return the provided default when conversion fails.
- Restored DEMUX dataset construction without bias weights by generating deterministic integer-lag pairs when scaled-time weights
  are unavailable.
