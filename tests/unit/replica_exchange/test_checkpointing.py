from unittest.mock import Mock, patch

from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


class TestReplicaExchangeCheckpointing:
    """Test checkpoint save/restore functionality."""

    def setup_remd(self, test_pdb_file, temp_output_dir):
        remd = ReplicaExchange(
            pdb_file=str(test_pdb_file),
            temperatures=[300, 310],
            output_dir=temp_output_dir,
            auto_setup=False,
        )

        remd._is_setup = True
        remd.contexts = [Mock(), Mock()]
        remd.replicas = [Mock(), Mock()]
        remd.exchange_attempts = 10
        remd.exchanges_accepted = 3
        remd.replica_states = [0, 1]
        remd.state_replicas = [0, 1]
        return remd

    def test_save_checkpoint_state(self, test_pdb_file, temp_output_dir):
        remd = self.setup_remd(test_pdb_file, temp_output_dir)
        state = remd.save_checkpoint_state()
        assert state["setup"] is True
        assert state["n_replicas"] == 2
        assert state["temperatures"] == [300, 310]
        assert state["exchange_attempts"] == 10
        assert state["exchanges_accepted"] == 3
        assert state["replica_states"] == [0, 1]
        assert state["state_replicas"] == [0, 1]

    def test_save_checkpoint_state_not_setup(self, test_pdb_file, temp_output_dir):
        remd = ReplicaExchange(
            pdb_file=str(test_pdb_file),
            temperatures=[300, 310],
            output_dir=temp_output_dir,
            auto_setup=False,
        )
        state = remd.save_checkpoint_state()
        assert state["setup"] is False

    def test_restore_from_checkpoint(self, test_pdb_file, temp_output_dir):
        remd = ReplicaExchange(
            pdb_file=str(test_pdb_file),
            temperatures=[300, 310],
            output_dir=temp_output_dir,
            auto_setup=False,
        )
        checkpoint_state = {
            "setup": True,
            "exchange_attempts": 15,
            "exchanges_accepted": 5,
            "replica_states": [1, 0],
            "state_replicas": [1, 0],
            "exchange_history": [[0, 1], [1, 0]],
        }
        with patch.object(remd, "setup_replicas") as mock_setup:
            remd.restore_from_checkpoint(checkpoint_state)
            mock_setup.assert_called_once()
        assert remd.exchange_attempts == 15
        assert remd.exchanges_accepted == 5
        assert remd.replica_states == [1, 0]
        assert remd.state_replicas == [1, 0]

    def test_rng_state_restored(self, test_pdb_file, temp_output_dir):
        remd = ReplicaExchange(
            pdb_file=str(test_pdb_file),
            temperatures=[300, 310],
            output_dir=temp_output_dir,
            auto_setup=False,
            random_seed=123,
        )
        remd._is_setup = True
        remd.contexts = [Mock(), Mock()]
        remd.replicas = [Mock(), Mock()]
        remd.integrators = [Mock(), Mock()]
        remd.rng.random()
        state = remd.save_checkpoint_state()
        expected_next = remd.rng.random()

        remd2 = ReplicaExchange(
            pdb_file=str(test_pdb_file),
            temperatures=[300, 310],
            output_dir=temp_output_dir,
            auto_setup=False,
            random_seed=999,
        )
        remd2._is_setup = True
        remd2.contexts = [Mock(), Mock()]
        remd2.replicas = [Mock(), Mock()]
        remd2.integrators = [Mock(), Mock()]
        remd2.restore_from_checkpoint(state)
        assert remd2.random_seed == 123
        assert remd2.rng.random() == expected_next
