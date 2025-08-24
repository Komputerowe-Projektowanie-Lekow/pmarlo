from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import mdtraj as md
import numpy as np
import pandas as pd

logger = logging.getLogger("pmarlo")


@dataclass
class CKTestResult:
    mse: Dict[int, float] = field(default_factory=dict)
    mode: str = "micro"  # or "macro"
    insufficient_data: bool = False
    thresholds: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]: ...


class MSMBase:
    def __init__(
        self,
        trajectory_files: Optional[Union[str, List[str]]] = None,
        topology_file: Optional[str] = None,
        temperatures: Optional[List[float]] = None,
        output_dir: str = "output/msm_analysis",
        random_state: Optional[int] = 42,
    ) -> None: ...

    def _update_total_frames(self) -> None: ...
