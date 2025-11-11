"""
Minimization policy strategies for replica exchange setup.

This module implements the Strategy Pattern for energy minimization:
- SinglePassPolicy: Minimize once, share state across all replicas (default, fastest)
- PerReplicaPolicy: Minimize each replica independently (safer for heterogeneous systems)

Feature detection automatically selects the appropriate policy based on system
characteristics (e.g., temperature-dependent parameters, replica-specific biases).

Trade-off: Increases abstraction complexity but guarantees correctness across
all REMD variants (standard T-REMD, Hamiltonian REMD, US-REMD, etc.).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MinimizationContext:
    """
    Context information for minimization decision-making.

    Attributes:
        num_replicas: Total number of replicas
        temperatures: Temperature ladder (K)
        has_per_replica_bias: Whether bias parameters vary by replica
        has_temperature_dependent_params: Whether system has T-dependent params
        system_forces: List of force objects from OpenMM System
        replica_index: Current replica being minimized
    """

    num_replicas: int
    temperatures: List[float]
    has_per_replica_bias: bool = False
    has_temperature_dependent_params: bool = False
    system_forces: Optional[List[Any]] = None
    replica_index: int = 0


class MinimizationPolicy(ABC):
    """
    Abstract base class for minimization strategies.

    Subclasses implement different trade-offs between performance and correctness
    for various REMD scenarios.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable policy name for logging."""
        pass

    @abstractmethod
    def should_minimize_replica(
        self, replica_index: int, context: MinimizationContext
    ) -> bool:
        """
        Determine if this replica needs independent minimization.

        Args:
            replica_index: Index of replica to check
            context: Decision-making context

        Returns:
            True if replica should be minimized independently
        """
        pass

    @abstractmethod
    def can_share_state(self) -> bool:
        """
        Whether this policy allows sharing minimized state between replicas.

        Returns:
            True if state sharing is allowed (SinglePass), False if not (PerReplica)
        """
        pass

    def get_performance_hint(self, context: MinimizationContext) -> str:
        """
        Return performance characteristics for user information.

        Args:
            context: Minimization context with system info
        """
        return "No specific performance hint available"


class SinglePassPolicy(MinimizationPolicy):
    """
    Minimize once, broadcast to all replicas (Compute-once, Broadcast Pattern).

    Optimal for standard T-REMD where all replicas share identical Hamiltonian
    at t=0 and only temperature differs during dynamics.

    Performance: O(1) minimizations regardless of replica count
    Safety: Only valid when system parameters are replica-independent
    """

    def get_name(self) -> str:
        return "SinglePass (shared state)"

    def should_minimize_replica(
        self, replica_index: int, context: MinimizationContext
    ) -> bool:
        """Only first replica (index 0) performs minimization."""
        return replica_index == 0

    def can_share_state(self) -> bool:
        return True

    def get_performance_hint(self, context: MinimizationContext) -> str:
        return (
            "O(1) minimizations - first replica computes, others reuse "
            f"(efficiency: {100 * (context.num_replicas - 1) / context.num_replicas:.1f}% "
            "computation saved)"
        )


class PerReplicaPolicy(MinimizationPolicy):
    """
    Minimize each replica independently.

    Required for:
    - Hamiltonian REMD (different force fields per replica)
    - Replica-specific biases (umbrella sampling REMD)
    - Temperature-dependent force field parameters

    Performance: O(N) minimizations where N = num_replicas
    Safety: Always correct, even for heterogeneous systems
    """

    def get_name(self) -> str:
        return "PerReplica (independent)"

    def should_minimize_replica(
        self, replica_index: int, context: MinimizationContext
    ) -> bool:
        """Every replica performs independent minimization."""
        return True

    def can_share_state(self) -> bool:
        return False

    def get_performance_hint(self, context: MinimizationContext) -> str:
        return (
            f"O(N) minimizations - {context.num_replicas} replicas each minimized "
            "independently (slower but safer for heterogeneous systems)"
        )


def detect_temperature_dependent_forces(system_forces: List[Any]) -> bool:
    """
    Detect if system has temperature-dependent force parameters.

    Some force fields (e.g., AMOEBA, custom temperature-dependent terms) may
    have parameters that change with temperature, making shared minimization unsafe.

    Args:
        system_forces: List of OpenMM Force objects

    Returns:
        True if temperature-dependent forces detected
    """
    if not system_forces:
        return False

    # Check for known temperature-dependent force types
    # (Extend this list based on force field requirements)
    temp_dependent_types = {
        "AmoebaVdwForce",  # AMOEBA has T-dependent vdW
        "AmoebaMultipoleForce",  # AMOEBA multipoles
        # Add other temperature-dependent forces as discovered
    }

    for force in system_forces:
        force_type = type(force).__name__
        if force_type in temp_dependent_types:
            logger.info(
                f"Detected temperature-dependent force: {force_type} "
                "(will use PerReplicaPolicy)"
            )
            return True

    return False


def detect_replica_specific_bias(
    bias_variables: Optional[List[Any]], num_replicas: int
) -> bool:
    """
    Detect if bias configuration varies by replica.

    For umbrella sampling REMD or other replica-dependent biasing, each replica
    may have different bias parameters requiring independent minimization.

    Args:
        bias_variables: Bias configuration (if any)
        num_replicas: Number of replicas

    Returns:
        True if per-replica biasing detected
    """
    if not bias_variables:
        return False

    # Check if bias_variables is a list of per-replica configurations
    if isinstance(bias_variables, list) and len(bias_variables) > 1:
        # Multiple bias configs suggest per-replica variation
        if len(bias_variables) == num_replicas:
            logger.info(
                "Detected per-replica bias configuration "
                f"({len(bias_variables)} configs for {num_replicas} replicas)"
            )
            return True

    return False


def auto_select_minimization_policy(
    num_replicas: int,
    temperatures: List[float],
    system_forces: Optional[List[Any]] = None,
    bias_variables: Optional[List[Any]] = None,
    force_policy: Optional[str] = None,
) -> MinimizationPolicy:
    """
    Automatically select optimal minimization policy via feature detection.

    Selection logic:
    1. If force_policy specified, use that (override)
    2. If replica-specific bias detected → PerReplicaPolicy
    3. If temperature-dependent forces detected → PerReplicaPolicy
    4. Otherwise → SinglePassPolicy (default, optimal for standard T-REMD)

    Args:
        num_replicas: Number of replicas
        temperatures: Temperature ladder
        system_forces: OpenMM Force objects for analysis
        bias_variables: Bias configuration (if any)
        force_policy: Manual override ("single" or "per_replica")

    Returns:
        Selected MinimizationPolicy instance

    Raises:
        ValueError: If force_policy is invalid
    """
    # Manual override
    if force_policy is not None:
        policy_lower = force_policy.lower()
        if policy_lower in ("single", "singlepass", "shared"):
            logger.info("Minimization policy: SinglePass (manual override)")
            return SinglePassPolicy()
        elif policy_lower in ("per_replica", "perreplica", "independent"):
            logger.info("Minimization policy: PerReplica (manual override)")
            return PerReplicaPolicy()
        else:
            raise ValueError(
                f"Invalid force_policy='{force_policy}'. "
                "Valid options: 'single', 'per_replica'"
            )

    # Feature detection
    has_replica_bias = detect_replica_specific_bias(bias_variables, num_replicas)
    has_temp_dependent = detect_temperature_dependent_forces(system_forces or [])

    # Decision tree
    if has_replica_bias:
        logger.info(
            "Auto-selected PerReplicaPolicy: per-replica bias detected "
            "(each replica has unique Hamiltonian)"
        )
        return PerReplicaPolicy()

    if has_temp_dependent:
        logger.info(
            "Auto-selected PerReplicaPolicy: temperature-dependent forces detected "
            "(minimized state varies by temperature)"
        )
        return PerReplicaPolicy()

    # Default: SinglePass for standard T-REMD (fastest)
    logger.info(
        "Auto-selected SinglePassPolicy: standard T-REMD detected "
        f"(will minimize once and share state across {num_replicas} replicas)"
    )
    return SinglePassPolicy()


def create_minimization_context(
    replica_index: int,
    num_replicas: int,
    temperatures: List[float],
    system_forces: Optional[List[Any]] = None,
    bias_variables: Optional[List[Any]] = None,
) -> MinimizationContext:
    """
    Create context object for minimization decisions.

    Args:
        replica_index: Current replica index
        num_replicas: Total number of replicas
        temperatures: Temperature ladder
        system_forces: OpenMM Force objects
        bias_variables: Bias configuration

    Returns:
        MinimizationContext with feature detection results
    """
    return MinimizationContext(
        num_replicas=num_replicas,
        temperatures=temperatures,
        has_per_replica_bias=detect_replica_specific_bias(bias_variables, num_replicas),
        has_temperature_dependent_params=detect_temperature_dependent_forces(
            system_forces or []
        ),
        system_forces=system_forces,
        replica_index=replica_index,
    )


def validate_policy_safety(
    policy: MinimizationPolicy, context: MinimizationContext
) -> None:
    """
    Validate that selected policy is safe for the given system.

    Raises warning if SinglePassPolicy used with potentially unsafe features.

    Args:
        policy: Selected minimization policy
        context: System context
    """
    if isinstance(policy, SinglePassPolicy):
        warnings = []

        if context.has_per_replica_bias:
            warnings.append("per-replica bias detected")

        if context.has_temperature_dependent_params:
            warnings.append("temperature-dependent force parameters")

        if warnings:
            logger.warning(
                f"SinglePassPolicy selected but {', '.join(warnings)} found. "
                "This may lead to incorrect minimization. Consider using "
                "PerReplicaPolicy or set force_policy='per_replica'."
            )
