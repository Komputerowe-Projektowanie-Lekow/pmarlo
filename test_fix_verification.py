"""Verify that the fixes for missing whitening metadata work correctly."""

import numpy as np
from pmarlo.analysis.fes import ensure_fes_inputs_whitened
from pmarlo.analysis.diagnostics import compute_diagnostics


def test_fes_whitening_without_metadata():
    """Test that FES whitening gracefully handles missing metadata."""
    # Dataset without __artifacts__
    dataset = {
        "X": np.random.randn(100, 2),
        "cv_names": ["cv1", "cv2"],
        "splits": {"train": {"X": np.random.randn(100, 2)}},
    }

    # Should return False (no whitening applied) instead of raising KeyError
    result = ensure_fes_inputs_whitened(dataset)
    assert result is False, "Expected False when no artifacts exist"
    print("✓ FES whitening handles missing __artifacts__")

    # Dataset with __artifacts__ but no mlcv_deeptica
    dataset_with_empty_artifacts = {
        "X": np.random.randn(100, 2),
        "cv_names": ["cv1", "cv2"],
        "__artifacts__": {},
        "splits": {"train": {"X": np.random.randn(100, 2)}},
    }

    result = ensure_fes_inputs_whitened(dataset_with_empty_artifacts)
    assert result is False, "Expected False when no mlcv_deeptica metadata exists"
    print("✓ FES whitening handles missing mlcv_deeptica metadata")


def test_diagnostics_without_metadata():
    """Test that diagnostics computation handles missing metadata."""
    X_train = np.random.randn(200, 2)
    X_val = np.random.randn(50, 2)

    # Dataset without metadata in splits
    dataset = {
        "splits": {
            "train": {"X": X_train},
            "val": {"X": X_val},
        }
    }

    # Should not raise ValueError/TypeError
    try:
        result = compute_diagnostics(dataset, diag_mass=0.5)
        assert "autocorrelation" in result
        assert "warnings" in result
        print("✓ Diagnostics computation handles missing whitening metadata")
        print(f"  - Found {len(result['autocorrelation'])} splits")
        print(f"  - Taus used: {result['taus']}")
    except (ValueError, TypeError) as e:
        print(f"✗ Diagnostics failed: {e}")
        raise


def test_fes_whitening_with_metadata():
    """Test that FES whitening still works when metadata is present."""
    coords = np.random.randn(100, 2)
    metadata = {
        "output_mean": np.zeros(2, dtype=np.float64),
        "output_transform": np.eye(2, dtype=np.float64),
        "output_transform_applied": False,
    }

    dataset = {
        "X": coords.copy(),
        "__artifacts__": {"mlcv_deeptica": metadata},
        "cv_names": ["cv1", "cv2"],
        "splits": {"train": {"X": coords.copy()}},
    }

    result = ensure_fes_inputs_whitened(dataset)
    assert result is True, "Expected True when whitening metadata exists"
    print("✓ FES whitening works correctly with metadata present")


if __name__ == "__main__":
    print("Testing fixes for missing whitening metadata...\n")

    test_fes_whitening_without_metadata()
    test_fes_whitening_with_metadata()
    test_diagnostics_without_metadata()

    print("\n✅ All tests passed! The fixes work correctly.")

