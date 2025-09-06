from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Feature flags and tuning parameters
DEMUX_STREAMING_ENABLED: bool = True
# Preferred key for selecting backend
DEMUX_BACKEND: str = "mdtraj"  # or "mdanalysis" if installed
# Backward-compat alias
DEMUX_IO_BACKEND: str = DEMUX_BACKEND
DEMUX_FILL_POLICY: str = "repeat"  # "repeat" | "skip" | "interpolate"
# Optional parallel segment readers; None disables parallelism
DEMUX_PARALLEL_WORKERS: int | None = None
# Chunk sizing: for mdtraj reader (chunk_size) and writer rewrite threshold
DEMUX_CHUNK_SIZE: int = 2048
# Force a writer flush after each segment when True
DEMUX_FLUSH_BETWEEN_SEGMENTS: bool = False
# Force a checkpoint flush every N segments (None disables)
DEMUX_CHECKPOINT_INTERVAL: int | None = None


@dataclass(frozen=True)
class RemdConfig:
    """Immutable configuration for REMD runs.

    This captures the knobs needed to construct and run replica-exchange.
    Keep runtime parameters immutable once a run starts.
    """

    pdb_file: str
    forcefield_files: List[str] = field(
        default_factory=lambda: ["amber14-all.xml", "amber14/tip3pfb.xml"]
    )
    temperatures: Optional[List[float]] = None
    output_dir: Path | str = Path("output/replica_exchange")
    exchange_frequency: int = 50
    dcd_stride: int = 1
    use_metadynamics: bool = True
    auto_setup: bool = False

    # Diagnostics/targets
    target_frames_per_replica: int = 5000
    target_accept: float = 0.30
    random_seed: Optional[int] = None
    # Resume options
    start_from_checkpoint: Optional[Path | str] = None
    start_from_pdb: Optional[Path | str] = None
    jitter_sigma_A: float = 0.0
