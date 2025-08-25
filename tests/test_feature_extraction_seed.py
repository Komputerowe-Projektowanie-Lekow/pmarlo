from unittest.mock import patch

import numpy as np

from pmarlo.simulation.simulation import feature_extraction


def test_feature_extraction_passes_random_state(
    test_trajectory_file, test_fixed_pdb_file
):
    """feature_extraction should forward the provided random_state to KMeans."""
    with patch("pmarlo.simulation.simulation.MiniBatchKMeans") as MBK:
        instance = MBK.return_value
        instance.fit.return_value = instance
        instance.labels_ = np.array([0])

        feature_extraction(
            str(test_trajectory_file),
            str(test_fixed_pdb_file),
            random_state=123,
        )

        MBK.assert_called_with(n_clusters=40, random_state=123)
        instance.fit.assert_called_once()
