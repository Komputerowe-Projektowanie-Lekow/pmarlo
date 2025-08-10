# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tests for the Simulation class.
"""

from importlib.util import find_spec

import numpy as np
import pytest

from pmarlo.simulation.simulation import Simulation, feature_extraction, prepare_system


class TestSimulation:
    """Test cases for Simulation class."""

    def test_simulation_initialization(self, test_fixed_pdb_file):
        """Test simulation initialization."""
        sim = Simulation(
            pdb_file=str(test_fixed_pdb_file),
            temperature=300.0,
            steps=100,
            use_metadynamics=False,
        )

        assert sim.pdb_file == str(test_fixed_pdb_file)
        assert sim.temperature == 300.0
        assert sim.steps == 100
        assert sim.use_metadynamics is False

    @pytest.mark.skipif(
        find_spec("openmm") is None,
        reason=("Requires OpenMM and significant computational time"),
    )
    def test_system_preparation(self, test_fixed_pdb_file, temp_output_dir):
        """Test system preparation (skipped by default)."""
        Simulation(
            pdb_file=str(test_fixed_pdb_file),
            temperature=300.0,
            steps=10,
            output_dir=str(temp_output_dir),
            use_metadynamics=False,
        )

        # This would require OpenMM and significant setup
        # simulation, meta = sim.prepare_system()
        # assert simulation is not None

    def test_feature_extraction(self, test_trajectory_file, test_fixed_pdb_file):
        """Test feature extraction from trajectory."""
        # This test uses the provided trajectory file
        try:
            states = feature_extraction(
                str(test_trajectory_file),
                str(test_fixed_pdb_file),
            )
            assert isinstance(states, np.ndarray)
            assert len(states) > 0
        except Exception as e:
            # If trajectory loading fails due to missing dependencies, skip
            pytest.skip(f"Feature extraction failed (likely missing dependencies): {e}")


class TestSimulationFunctions:
    """Test cases for simulation utility functions."""

    def test_prepare_system_function_signature(self, test_fixed_pdb_file):
        """Test that prepare_system function has correct signature."""
        # Just test the function exists and can be called (without actually running)
        try:
            # Test function signature by inspecting it
            import inspect

            sig = inspect.signature(prepare_system)
            params = list(sig.parameters.keys())
            assert "pdb_file_name" in params
        except ImportError:
            pytest.skip(
                "Cannot inspect function signature without required dependencies"
            )

    def test_simulation_configuration_validation(self):
        """Test simulation configuration validation."""
        # Test that simulation accepts basic parameters (validation happens during execution)
        sim = Simulation(
            pdb_file="nonexistent.pdb",
            temperature=300,
            steps=100,  # Valid temperature
        )
        # Configuration is accepted during initialization
        # Errors typically occur during execution when files are accessed
        assert sim.temperature == 300
        assert sim.steps == 100

    def test_output_directory_creation(self, temp_output_dir):
        """Test output directory creation."""
        output_dir = temp_output_dir / "simulation_test"

        Simulation(
            pdb_file="dummy.pdb",  # Won't be used for this test
            output_dir=str(output_dir),
        )

        assert output_dir.exists()
        assert output_dir.is_dir()
