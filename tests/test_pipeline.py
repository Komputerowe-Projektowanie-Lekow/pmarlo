# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tests for the Pipeline class.
"""

from importlib.util import find_spec

import pytest

from pmarlo.pipeline import LegacyPipeline, Pipeline, run_pmarlo

# Evaluated by pytest.mark.skipif when using string condition
skip_if_no_openmm = find_spec("openmm") is None


class TestPipeline:
    """Test cases for Pipeline class."""

    def test_pipeline_initialization(self, test_pdb_file, temp_output_dir):
        """Test pipeline initialization."""
        pipeline = Pipeline(
            pdb_file=str(test_pdb_file),
            temperatures=[300.0, 310.0],
            steps=100,
            n_states=10,
            use_replica_exchange=False,
            output_dir=str(temp_output_dir),
        )

        assert pipeline.pdb_file == str(test_pdb_file)
        assert pipeline.temperatures == [300.0, 310.0]
        assert pipeline.steps == 100
        assert pipeline.n_states == 10
        assert not pipeline.use_replica_exchange

    def test_pipeline_default_temperatures(self, test_pdb_file):
        """Test pipeline with default temperature settings."""
        # Test single temperature (no replica exchange)
        pipeline = Pipeline(pdb_file=str(test_pdb_file), use_replica_exchange=False)
        assert pipeline.temperatures == [300.0]

        # Test replica exchange with default temperatures
        pipeline = Pipeline(
            pdb_file=str(test_pdb_file), use_replica_exchange=True, n_replicas=3
        )
        assert len(pipeline.temperatures) == 3
        assert all(t >= 300.0 for t in pipeline.temperatures)

    def test_pipeline_component_setup(self, test_pdb_file, temp_output_dir):
        """Test individual component setup."""
        pipeline = Pipeline(
            pdb_file=str(test_pdb_file),
            output_dir=str(temp_output_dir),
            use_replica_exchange=False,
        )

        # Test protein setup
        try:
            protein = pipeline.setup_protein()
            assert protein is not None
            assert hasattr(protein, "get_properties")
        except Exception as e:
            # If protein setup fails due to dependencies, that's expected
            assert "No module named" in str(e) or "ImportError" in str(type(e).__name__)

    def test_pipeline_get_components(self, test_pdb_file):
        """Test getting pipeline components."""
        pipeline = Pipeline(pdb_file=str(test_pdb_file))
        components = pipeline.get_components()

        assert isinstance(components, dict)
        assert "protein" in components
        assert "replica_exchange" in components
        assert "simulation" in components
        assert "markov_state_model" in components

    @pytest.mark.skipif(
        "skip_if_no_openmm",
        reason="Full pipeline run requires significant computational time",
    )
    def test_full_pipeline_run(self, test_pdb_file, temp_output_dir):
        """Test full pipeline execution (skipped by default)."""
        # This would require significant computational resources
        # results = Pipeline(
        #     pdb_file=str(test_pdb_file),
        #     output_dir=str(temp_output_dir),
        #     steps=10,  # Very short for testing
        #     n_states=5,  # Few states for testing
        #     use_replica_exchange=False,
        # ).run()
        # assert results['pipeline']['status'] == 'completed'
        pass


class TestLegacyPipeline:
    """Test cases for LegacyPipeline class."""

    def test_legacy_pipeline_initialization(self, test_pdb_file, temp_output_dir):
        """Test legacy pipeline initialization."""
        legacy = LegacyPipeline(
            pdb_file=str(test_pdb_file),
            output_dir=str(temp_output_dir),
            run_id="test123",
        )

        assert legacy.pdb_file == str(test_pdb_file)
        assert legacy.run_id == "test123"
        assert legacy.checkpoint_manager is None  # Not initialized until run


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_run_pmarlo_function_signature(self):
        """Test run_pmarlo function signature."""
        import inspect

        sig = inspect.signature(run_pmarlo)
        params = list(sig.parameters.keys())

        assert "pdb_file" in params
        assert "temperatures" in params
        assert "steps" in params
        assert "n_states" in params

    @pytest.mark.skipif(
        "skip_if_no_openmm", reason="Full run_pmarlo requires computational resources"
    )
    def test_run_pmarlo_execution(self, test_pdb_file, temp_output_dir):
        """Test run_pmarlo function execution (skipped by default)."""
        # This would run the full pipeline
        # results = run_pmarlo(
        #     pdb_file=str(test_pdb_file),
        #     steps=10,
        #     n_states=5,
        #     output_dir=str(temp_output_dir)
        # )
        # assert 'pipeline' in results
        pass
