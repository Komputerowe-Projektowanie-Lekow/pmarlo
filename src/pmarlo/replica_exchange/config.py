from __future__ import annotations

import warnings
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

    pdb_file: str | Path | None = None
    input_pdb: str | Path | None = None
    forcefield_files: List[str] = field(
        default_factory=lambda: ["amber14-all.xml", "amber14/tip3pfb.xml"]
    )
    temperatures: Optional[List[float]] = None
    output_dir: Path | str | None = None
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
    reseed_velocities: bool = False
    temperature_schedule_mode: str | None = None

    def __post_init__(self) -> None:
        resolved = self.pdb_file if self.pdb_file is not None else self.input_pdb
        if resolved is None:
            raise TypeError("RemdConfig requires `pdb_file` or `input_pdb`.")
        resolved_str = str(resolved)
        object.__setattr__(self, "pdb_file", resolved_str)
        if self.input_pdb is not None:
            if str(self.input_pdb) != resolved_str:
                warnings.warn(
                    "`input_pdb` differs from `pdb_file`; using `input_pdb` value.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                resolved_str = str(self.input_pdb)
                object.__setattr__(self, "pdb_file", resolved_str)
            warnings.warn(
                "`input_pdb` is deprecated and will be removed in PMARLO 0.3. Pass `pdb_file` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(self, "input_pdb", resolved_str)
        else:
            object.__setattr__(self, "input_pdb", resolved_str)

        if self.output_dir is None:
            raise TypeError(
                "RemdConfig requires `output_dir` to be provided explicitly."
            )
        object.__setattr__(self, "output_dir", Path(self.output_dir))
