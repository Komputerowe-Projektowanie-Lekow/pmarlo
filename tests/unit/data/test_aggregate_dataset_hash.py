from __future__ import annotations

import numpy as np

from pmarlo.data.aggregate import _dataset_hash


def test_dataset_hash_changes_with_periodic_flags() -> None:
    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    dtrajs = [np.arange(6, dtype=np.int32)]
    cv_names = ("phi", "psi")

    hash_periodic = _dataset_hash(dtrajs, X, cv_names, (True, False))
    hash_non_periodic = _dataset_hash(dtrajs, X, cv_names, (False, False))

    assert hash_periodic != hash_non_periodic
