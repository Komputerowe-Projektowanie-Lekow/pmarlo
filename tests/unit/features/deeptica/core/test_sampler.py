from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SAMPLER_MODULE = (
    ROOT / "src" / "pmarlo" / "features" / "deeptica_trainer" / "sampler.py"
)

spec = importlib.util.spec_from_file_location("_pmarlo_sampler", SAMPLER_MODULE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from pmarlo.pairs.core import PairInfo  # noqa: E402

iter_pair_batches = module.iter_pair_batches


def test_balanced_temp_sampler_yields_complete_cover():
    pair_info = PairInfo(
        idx_t=np.arange(10, dtype=np.int64),
        idx_tau=np.arange(10, dtype=np.int64) + 1,
        weights=np.ones(10, dtype=np.float32),
        diagnostics={"pairs_by_shard": [10]},
    )
    batches = list(iter_pair_batches(pair_info, batch_size=4, seed=123))
    concatenated = np.concatenate(batches)
    assert np.unique(concatenated).size == pair_info.idx_t.size
