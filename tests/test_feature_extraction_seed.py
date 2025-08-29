from unittest.mock import patch

import numpy as np

from pmarlo.simulation.simulation import feature_extraction


def test_feature_extraction_passes_random_state(
    test_trajectory_file, test_fixed_pdb_file
):
    """feature_extraction should forward the provided random_state."""
    with patch("pmarlo.api.cluster_microstates") as cm:
        cm.return_value = np.array([0])

        feature_extraction(
            str(test_trajectory_file),
            str(test_fixed_pdb_file),
            random_state=123,
        )

        assert cm.call_args.kwargs["random_state"] == 123
        assert cm.call_args.kwargs["n_states"] == 40
