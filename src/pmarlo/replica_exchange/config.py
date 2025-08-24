from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
