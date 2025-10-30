from __future__ import annotations

import numpy as np
import pytest

from pmarlo.analysis.project_cv import apply_whitening_from_metadata


def test_apply_whitening_requires_mapping() -> None:
    values = np.ones((4, 2), dtype=np.float64)

    with pytest.raises(TypeError):
        apply_whitening_from_metadata(values, metadata=object())


def test_apply_whitening_applies_transform_and_sets_flag() -> None:
    values = np.array([[2.0, 5.0], [3.0, 7.0]], dtype=np.float64)
    metadata: dict[str, object] = {
        "output_mean": [1.0, 4.0],
        "output_transform": np.eye(2, dtype=np.float64),
        "output_transform_applied": False,
    }

    whitened, applied = apply_whitening_from_metadata(values, metadata)

    assert applied is True
    np.testing.assert_allclose(
        whitened, np.array([[-0.5, -1.0], [0.5, 1.0]], dtype=np.float64)
    )
    assert metadata["output_transform_applied"] is True


def test_apply_whitening_raises_when_metadata_missing() -> None:
    values = np.array([[2.0, 5.0]], dtype=np.float64)
    metadata = {"output_mean": None, "output_transform": None}

    with pytest.raises(ValueError):
        apply_whitening_from_metadata(values, metadata)
