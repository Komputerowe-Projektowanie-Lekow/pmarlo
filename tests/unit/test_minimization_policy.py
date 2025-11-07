"""
Unit tests for minimization policy strategy pattern.

Tests the Strategy Pattern implementation for replica exchange minimization,
including auto-selection logic and feature detection.
"""

import pytest
from pmarlo.replica_exchange.minimization_policy import (
    MinimizationContext,
    MinimizationPolicy,
    SinglePassPolicy,
    PerReplicaPolicy,
    auto_select_minimization_policy,
    create_minimization_context,
    detect_temperature_dependent_forces,
    detect_replica_specific_bias,
    validate_policy_safety,
)


class TestMinimizationPolicies:
    """Test individual policy implementations."""

    def test_single_pass_policy_basic(self):
        """SinglePassPolicy should minimize only first replica."""
        policy = SinglePassPolicy()
        context = MinimizationContext(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
        )

        assert policy.get_name() == "SinglePass (shared state)"
        assert policy.can_share_state() is True
        assert policy.should_minimize_replica(0, context) is True
        assert policy.should_minimize_replica(1, context) is False
        assert policy.should_minimize_replica(2, context) is False
        assert policy.should_minimize_replica(3, context) is False

    def test_single_pass_performance_hint(self):
        """SinglePassPolicy should report correct efficiency."""
        policy = SinglePassPolicy()
        context = MinimizationContext(
            num_replicas=8,
            temperatures=[300.0] * 8,
        )

        hint = policy.get_performance_hint(context)
        assert "O(1)" in hint
        assert "87.5%" in hint  # (8-1)/8 = 87.5%

    def test_per_replica_policy_basic(self):
        """PerReplicaPolicy should minimize all replicas."""
        policy = PerReplicaPolicy()
        context = MinimizationContext(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
        )

        assert policy.get_name() == "PerReplica (independent)"
        assert policy.can_share_state() is False
        assert policy.should_minimize_replica(0, context) is True
        assert policy.should_minimize_replica(1, context) is True
        assert policy.should_minimize_replica(2, context) is True
        assert policy.should_minimize_replica(3, context) is True

    def test_per_replica_performance_hint(self):
        """PerReplicaPolicy should report O(N) scaling."""
        policy = PerReplicaPolicy()
        context = MinimizationContext(
            num_replicas=5,
            temperatures=[300.0] * 5,
        )

        hint = policy.get_performance_hint(context)
        assert "O(N)" in hint
        assert "5 replicas" in hint


class TestFeatureDetection:
    """Test automatic feature detection logic."""

    def test_detect_temperature_dependent_forces_empty(self):
        """Empty force list should return False."""
        assert detect_temperature_dependent_forces([]) is False
        assert detect_temperature_dependent_forces(None) is False

    def test_detect_temperature_dependent_forces_amoeba(self):
        """Should detect AMOEBA forces as temperature-dependent."""
        # Mock AMOEBA force
        class AmoebaVdwForce:
            pass

        forces = [AmoebaVdwForce()]
        assert detect_temperature_dependent_forces(forces) is True

    def test_detect_temperature_dependent_forces_standard(self):
        """Standard forces should not be flagged as temperature-dependent."""
        # Mock standard OpenMM forces
        class HarmonicBondForce:
            pass

        class NonbondedForce:
            pass

        forces = [HarmonicBondForce(), NonbondedForce()]
        assert detect_temperature_dependent_forces(forces) is False

    def test_detect_replica_specific_bias_none(self):
        """No bias should return False."""
        assert detect_replica_specific_bias(None, 4) is False
        assert detect_replica_specific_bias([], 4) is False

    def test_detect_replica_specific_bias_single_config(self):
        """Single bias config should return False (shared across replicas)."""
        bias = [{"center": 1.0, "kappa": 100.0}]
        assert detect_replica_specific_bias(bias, 4) is False

    def test_detect_replica_specific_bias_per_replica(self):
        """Per-replica bias configs should be detected."""
        # One config per replica (umbrella sampling)
        bias = [
            {"center": 0.5, "kappa": 100.0},
            {"center": 1.0, "kappa": 100.0},
            {"center": 1.5, "kappa": 100.0},
            {"center": 2.0, "kappa": 100.0},
        ]
        assert detect_replica_specific_bias(bias, 4) is True


class TestAutoSelection:
    """Test automatic policy selection logic."""

    def test_auto_select_default_single_pass(self):
        """Standard T-REMD should default to SinglePassPolicy."""
        policy = auto_select_minimization_policy(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            system_forces=[],
            bias_variables=None,
        )

        assert isinstance(policy, SinglePassPolicy)

    def test_auto_select_manual_override_single(self):
        """Manual override should select SinglePassPolicy."""
        policy = auto_select_minimization_policy(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            force_policy="single",
        )
        assert isinstance(policy, SinglePassPolicy)

        # Test alternative names
        policy = auto_select_minimization_policy(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            force_policy="singlepass",
        )
        assert isinstance(policy, SinglePassPolicy)

    def test_auto_select_manual_override_per_replica(self):
        """Manual override should select PerReplicaPolicy."""
        policy = auto_select_minimization_policy(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            force_policy="per_replica",
        )
        assert isinstance(policy, PerReplicaPolicy)

        # Test alternative names
        policy = auto_select_minimization_policy(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            force_policy="independent",
        )
        assert isinstance(policy, PerReplicaPolicy)

    def test_auto_select_invalid_policy_name(self):
        """Invalid policy name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid force_policy"):
            auto_select_minimization_policy(
                num_replicas=4,
                temperatures=[300.0, 310.0, 320.0, 330.0],
                force_policy="invalid_name",
            )

    def test_auto_select_with_replica_bias(self):
        """Per-replica bias should trigger PerReplicaPolicy."""
        bias = [
            {"center": 0.5},
            {"center": 1.0},
            {"center": 1.5},
            {"center": 2.0},
        ]

        policy = auto_select_minimization_policy(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            bias_variables=bias,
        )

        assert isinstance(policy, PerReplicaPolicy)

    def test_auto_select_with_temp_dependent_forces(self):
        """Temperature-dependent forces should trigger PerReplicaPolicy."""
        class AmoebaVdwForce:
            pass

        forces = [AmoebaVdwForce()]

        policy = auto_select_minimization_policy(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            system_forces=forces,
        )

        assert isinstance(policy, PerReplicaPolicy)


class TestMinimizationContext:
    """Test MinimizationContext creation and usage."""

    def test_create_context_basic(self):
        """Basic context creation should work."""
        context = create_minimization_context(
            replica_index=2,
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
        )

        assert context.replica_index == 2
        assert context.num_replicas == 4
        assert len(context.temperatures) == 4
        assert context.has_per_replica_bias is False
        assert context.has_temperature_dependent_params is False

    def test_create_context_with_features(self):
        """Context should detect features automatically."""
        bias = [{"center": i} for i in range(4)]

        class AmoebaVdwForce:
            pass

        forces = [AmoebaVdwForce()]

        context = create_minimization_context(
            replica_index=1,
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            system_forces=forces,
            bias_variables=bias,
        )

        assert context.has_per_replica_bias is True
        assert context.has_temperature_dependent_params is True


class TestPolicySafety:
    """Test policy safety validation."""

    def test_validate_single_pass_safe(self):
        """SinglePassPolicy with no special features should be safe (no warnings)."""
        policy = SinglePassPolicy()
        context = MinimizationContext(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            has_per_replica_bias=False,
            has_temperature_dependent_params=False,
        )

        # Should not raise or warn (test passes if no exception)
        validate_policy_safety(policy, context)

    def test_validate_single_pass_with_bias(self, caplog):
        """SinglePassPolicy with per-replica bias should warn."""
        import logging

        policy = SinglePassPolicy()
        context = MinimizationContext(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            has_per_replica_bias=True,
            has_temperature_dependent_params=False,
        )

        with caplog.at_level(logging.WARNING):
            validate_policy_safety(policy, context)

        # Check that warning was logged
        assert any("per-replica bias" in record.message for record in caplog.records)

    def test_validate_single_pass_with_temp_dependent(self, caplog):
        """SinglePassPolicy with T-dependent forces should warn."""
        import logging

        policy = SinglePassPolicy()
        context = MinimizationContext(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            has_per_replica_bias=False,
            has_temperature_dependent_params=True,
        )

        with caplog.at_level(logging.WARNING):
            validate_policy_safety(policy, context)

        # Check that warning was logged
        assert any("temperature-dependent" in record.message for record in caplog.records)

    def test_validate_per_replica_always_safe(self):
        """PerReplicaPolicy should always be safe (no warnings)."""
        policy = PerReplicaPolicy()
        context = MinimizationContext(
            num_replicas=4,
            temperatures=[300.0, 310.0, 320.0, 330.0],
            has_per_replica_bias=True,
            has_temperature_dependent_params=True,
        )

        # Should not warn even with special features
        validate_policy_safety(policy, context)


class TestPolicyIntegration:
    """Integration tests for policy usage scenarios."""

    def test_standard_tremd_workflow(self):
        """Simulate standard T-REMD setup workflow."""
        # Standard T-REMD: no bias, standard forces
        policy = auto_select_minimization_policy(
            num_replicas=8,
            temperatures=[300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0],
            system_forces=[],
            bias_variables=None,
        )

        context = create_minimization_context(
            replica_index=0,
            num_replicas=8,
            temperatures=[300.0, 310.0, 320.0, 330.0, 340.0, 350.0, 360.0, 370.0],
        )

        # Should use SinglePassPolicy for efficiency
        assert isinstance(policy, SinglePassPolicy)
        assert policy.should_minimize_replica(0, context) is True
        for i in range(1, 8):
            context.replica_index = i
            assert policy.should_minimize_replica(i, context) is False

    def test_umbrella_sampling_remd_workflow(self):
        """Simulate umbrella sampling REMD setup workflow."""
        # US-REMD: per-replica bias centers
        num_replicas = 6
        bias = [{"center": i * 0.5, "kappa": 100.0} for i in range(num_replicas)]

        policy = auto_select_minimization_policy(
            num_replicas=num_replicas,
            temperatures=[300.0] * num_replicas,
            bias_variables=bias,
        )

        context = create_minimization_context(
            replica_index=0,
            num_replicas=num_replicas,
            temperatures=[300.0] * num_replicas,
            bias_variables=bias,
        )

        # Should use PerReplicaPolicy for correctness
        assert isinstance(policy, PerReplicaPolicy)
        for i in range(num_replicas):
            context.replica_index = i
            assert policy.should_minimize_replica(i, context) is True

