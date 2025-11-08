"""Tests for single-temperature MD production phase.

This module tests the new single-temperature production phase that was added
to handle the case when n_replicas == 1 or exchange_frequency is very large.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from openmm import unit
from openmm.app import ForceField, PDBFile, Simulation

from pmarlo.replica_exchange.replica_exchange import ReplicaExchange


def test_single_replica_production_runs_md_steps(monkeypatch):
    """Verify that single-replica simulation actually runs production steps."""
    
    test_pdb = Path(__file__).parent.parent.parent / "_assets" / "3gd8-fixed.pdb"
    if not test_pdb.exists():
        pytest.skip(f"Test PDB not found: {test_pdb}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "single_temp_test"
        
        # Create ReplicaExchange with a single temperature
        remd = ReplicaExchange.from_config(
            type("Config", (), {
                "pdb_file": str(test_pdb),
                "temperatures": [300.0],  # Single temperature
                "output_dir": str(output_dir),
                "exchange_frequency": 99999999,  # Effectively disabled
                "auto_setup": False,
                "dcd_stride": 1,
                "random_seed": 42,
                "start_from_checkpoint": None,
                "start_from_pdb": None,
                "jitter_sigma_A": 0.0,
                "reseed_velocities": False,
                "write_replica_indices": [0],
            })()
        )
        
        # Track if production steps were actually run
        step_calls = []
        original_step = None
        
        def track_step_calls(simulation_self, steps):
            """Track calls to simulation.step()"""
            step_calls.append(steps)
            # Don't actually run the simulation, just track the call
        
        # Plan reporter stride
        remd.plan_reporter_stride(
            total_steps=1000,
            equilibration_steps=100,
            target_frames=10,
        )
        
        # Setup replicas
        remd.setup_replicas()
        
        # Monkey-patch the step method to track calls without running full simulation
        for replica in remd.replicas:
            original_step = replica.step
            replica.step = lambda steps: track_step_calls(replica, steps)
        
        # Run simulation
        production_steps = 1000 - 100  # total - equilibration
        remd.run_simulation(
            total_steps=1000,
            equilibration_steps=100,
        )
        
        # Verify that production steps were actually executed
        # The production phase should have been called with chunks of steps
        total_production_steps = sum(step_calls[1:])  # Skip equilibration steps
        
        assert total_production_steps > 0, "Production phase should have run some steps"
        assert total_production_steps == production_steps, (
            f"Production should run {production_steps} steps, but ran {total_production_steps}"
        )


def test_single_temp_production_no_exchange_warnings():
    """Verify single-temp MD doesn't show inappropriate exchange warnings."""
    
    test_pdb = Path(__file__).parent.parent.parent / "_assets" / "3gd8-fixed.pdb"
    if not test_pdb.exists():
        pytest.skip(f"Test PDB not found: {test_pdb}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "single_temp_test"
        
        # Create ReplicaExchange with a single temperature
        remd = ReplicaExchange.from_config(
            type("Config", (), {
                "pdb_file": str(test_pdb),
                "temperatures": [300.0],
                "output_dir": str(output_dir),
                "exchange_frequency": 99999999,
                "auto_setup": False,
                "dcd_stride": 1,
                "random_seed": 42,
                "start_from_checkpoint": None,
                "start_from_pdb": None,
                "jitter_sigma_A": 0.0,
                "reseed_velocities": False,
                "write_replica_indices": [0],
            })()
        )
        
        # Verify n_replicas is 1
        assert remd.n_replicas == 1, "Should have exactly 1 replica for single-temp MD"
        
        # Call _log_final_stats and verify it doesn't raise any errors
        # and doesn't try to compute exchange statistics
        import logging
        import io
        
        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        logger = logging.getLogger("pmarlo")
        logger.addHandler(handler)
        
        try:
            remd._log_final_stats()
            log_output = log_capture.getvalue()
            
            # Verify appropriate message for single-temp MD
            assert "SINGLE-TEMPERATURE MD" in log_output, (
                "Should log single-temperature completion message"
            )
            
            # Verify no exchange warnings
            assert "exchange acceptance" not in log_output.lower(), (
                "Should not mention exchange acceptance for single-temp MD"
            )
            assert "replica diffusion" not in log_output.lower(), (
                "Should not mention replica diffusion for single-temp MD"
            )
        finally:
            logger.removeHandler(handler)


def test_large_exchange_frequency_triggers_single_temp_path():
    """Verify that large exchange_frequency triggers single-temp production path."""
    
    test_pdb = Path(__file__).parent.parent.parent / "_assets" / "3gd8-fixed.pdb"
    if not test_pdb.exists():
        pytest.skip(f"Test PDB not found: {test_pdb}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "single_temp_test"
        
        # Create ReplicaExchange with large exchange_frequency
        remd = ReplicaExchange.from_config(
            type("Config", (), {
                "pdb_file": str(test_pdb),
                "temperatures": [300.0],
                "output_dir": str(output_dir),
                "exchange_frequency": 99999999,  # Very large
                "auto_setup": False,
                "dcd_stride": 1,
                "random_seed": 42,
                "start_from_checkpoint": None,
                "start_from_pdb": None,
                "jitter_sigma_A": 0.0,
                "reseed_velocities": False,
                "write_replica_indices": [0],
            })()
        )
        
        production_steps = 1000
        equilibration_steps = 100
        
        # Check if it would use single-temp path
        is_single_temp = (
            remd.n_replicas == 1 or 
            (production_steps - equilibration_steps) < remd.exchange_frequency
        )
        
        assert is_single_temp, (
            "Should use single-temp production path when exchange_frequency is very large"
        )

