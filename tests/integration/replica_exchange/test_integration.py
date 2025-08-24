from pathlib import Path
from unittest.mock import Mock, patch

from pmarlo.pipeline import Pipeline


class TestReplicaExchangeIntegration:
    """Integration tests for full workflow."""

    def test_pipeline_integration(self, test_fixed_pdb_file, temp_output_dir):
        pipeline = Pipeline(
            pdb_file=str(test_fixed_pdb_file),
            temperatures=[300, 310],
            steps=100,
            output_dir=temp_output_dir,
            use_replica_exchange=True,
            use_metadynamics=False,
        )
        with patch.object(pipeline, "setup_protein") as mock_protein:
            mock_protein.return_value = Mock()
            mock_protein.return_value.get_properties.return_value = {
                "num_atoms": 100,
                "num_residues": 10,
            }
            pipeline.prepared_pdb = Path(test_fixed_pdb_file)
            remd = pipeline.setup_replica_exchange()
            assert remd is not None
            assert remd.is_setup()
