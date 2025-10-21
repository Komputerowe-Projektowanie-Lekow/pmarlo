from __future__ import annotations

import numpy as np
import pytest

from pmarlo.analysis.validation import ValidationError, validate_features


def test_validate_features_constant_column_raises_zero_std() -> None:
    X = np.column_stack(
        (
            np.ones(8, dtype=np.float64),
            np.linspace(-1.0, 1.0, 8, dtype=np.float64),
        )
    )

    with pytest.raises(ValidationError) as excinfo:
        validate_features(X, ["constant", "varying"])

    error = excinfo.value
    assert error.code == "cv_zero_std"
    assert "problematic_features" in error.stats
    assert "constant" in error.stats["problematic_features"]


def test_validate_features_no_finite_rows_raises() -> None:
    X = np.array(
        [
            [np.nan, np.nan],
            [np.inf, -np.inf],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValidationError) as excinfo:
        validate_features(X, ["a", "b"])

    assert excinfo.value.code == "cv_no_finite_rows"


def test_validate_features_success_returns_stats() -> None:
    X = np.array(
        [
            [-1.0, 0.5],
            [0.0, 1.5],
            [1.0, -0.5],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )

    stats = validate_features(X, ["cv1", "cv2"])

    assert stats["finite_rows"] == 4
    assert stats["n_features"] == 2
    assert stats["feature_names"] == ["cv1", "cv2"]
    assert all(std > 0 for std in stats["stds"])
