"""Test biased simulation visualization functionality."""

import numpy as np
import pytest


def test_biased_visualization_basic():
    """Test that biased simulations are visually distinguishable in validation plots."""
    from pmarlo.visualization.diagnostics import create_sampling_validation_plot

    # Create synthetic trajectory data
    np.random.seed(42)

    # Standard (unbiased) trajectory - more exploration
    traj_standard_1 = np.random.randn(1000, 2) * 2.0
    traj_standard_1[:, 0] += np.linspace(-2, 2, 1000)  # Linear transition

    # Another standard trajectory
    traj_standard_2 = np.random.randn(800, 2) * 1.5
    traj_standard_2[:, 0] += np.linspace(-1, 3, 800)

    # Biased trajectory - more focused sampling
    traj_biased_1 = np.random.randn(1200, 2) * 1.0
    traj_biased_1[:, 0] += 1.0  # Centered around 1.0

    # Another biased trajectory
    traj_biased_2 = np.random.randn(900, 2) * 1.2
    traj_biased_2[:, 0] += np.sin(np.linspace(0, 4*np.pi, 900)) * 2

    projection_data = [
        traj_standard_1,
        traj_biased_1,
        traj_standard_2,
        traj_biased_2,
    ]

    run_labels = [
        "run-001-standard",
        "run-002-biased",
        "run-003-standard",
        "run-004-biased",
    ]

    metabiased_runs = [False, True, False, True]

    fig = create_sampling_validation_plot(
        projection_data=projection_data,
        run_labels=run_labels,
        metabiased_runs=metabiased_runs,
        max_length=1000,
        hist_bins=100,
        stride=5,
    )

    # Check that the plot was created
    assert fig is not None
    assert len(fig.axes) > 0

    # Get legend
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None

    # Check legend labels
    legend_labels = [t.get_text() for t in legend.get_texts()]

    # Verify biased runs are marked
    assert any("biased" in label.lower() for label in legend_labels), \
        "Expected 'biased' or 'metabiased' in legend labels"

    # Check for style indicators in legend title
    legend_title = legend.get_title().get_text()
    assert "dashed" in legend_title.lower() or "metabiased" in legend_title.lower(), \
        "Expected legend title to indicate dashed lines for metabiased runs"

    # Check that we have both standard and metabiased style handles
    assert "Standard Run" in legend_labels or "standard" in legend_title.lower()
    assert "Metabiased Run" in legend_labels or "metabiased" in legend_title.lower()

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_biased_visualization_empty_lists():
    """Test that function handles empty inputs gracefully."""
    from pmarlo.visualization.diagnostics import create_sampling_validation_plot

    # Test with empty lists
    with pytest.raises((ValueError, AssertionError)):
        create_sampling_validation_plot(
            projection_data=[],
            run_labels=[],
            metabiased_runs=[],
        )


def test_biased_visualization_mismatched_lengths():
    """Test that function validates input list lengths."""
    from pmarlo.visualization.diagnostics import create_sampling_validation_plot

    np.random.seed(42)
    traj1 = np.random.randn(100, 2)

    # Mismatched lengths should raise an error
    with pytest.raises((ValueError, AssertionError)):
        create_sampling_validation_plot(
            projection_data=[traj1],
            run_labels=["run1", "run2"],  # Wrong length
            metabiased_runs=[False],
        )
