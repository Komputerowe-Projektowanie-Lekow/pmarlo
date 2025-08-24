from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import mdtraj as md


class LoadingMixin:
    def load_trajectories(
        self,
        *,
        stride: int = 1,
        atom_selection: str | Sequence[int] | None = None,
        chunk_size: int = 1000,
    ) -> None: ...

    def _resolve_atom_indices(
        self, atom_selection: str | Sequence[int] | None
    ) -> Sequence[int] | None: ...

    def _stream_single_trajectory(
        self,
        *,
        traj_file: str,
        stride: int,
        atom_indices: Sequence[int] | None,
        chunk_size: int,
        selection_str: str | None,
    ) -> md.Trajectory | None: ...

    def _maybe_load_demux_metadata(self, traj_path: Path) -> None: ...
