from pathlib import Path

import numpy as np
from openmm.app import PDBFile

from pmarlo.replica_exchange.running_stats import RunningStats

ASSET_PDB = Path("tests/_assets/3gd8-fixed.pdb")


def _load_real_positions() -> np.ndarray:
    pdb = PDBFile(str(ASSET_PDB))
    return np.asarray(pdb.getPositions(asNumpy=True)._value)


def test_running_stats_preserves_float32_dtype_with_real_positions() -> None:
    positions = _load_real_positions().astype(np.float32)
    stats = RunningStats(dim=3, dtype=np.float32)
    for xyz in positions[:32]:
        stats.update(xyz)

    mean, std = stats.summary(copy=True)
    assert mean.dtype == np.float32
    assert std.dtype == np.float32
    assert stats.count == 32


def test_running_stats_defaults_to_float64_for_openmm_data() -> None:
    positions = _load_real_positions()
    stats = RunningStats(dim=3)
    for xyz in positions[:16]:
        stats.update(xyz)

    mean, std = stats.summary()
    assert mean.dtype == np.float64
    assert std.dtype == np.float64
    assert stats.count == 16
