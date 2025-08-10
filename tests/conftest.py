# Copyright (c) 2025 PMARLO Development Team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Pytest configuration and fixtures for PMARLO tests.
"""

import shutil
import tempfile
from importlib.util import find_spec
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "pdbfixer: mark test as requiring PDBFixer",
    )


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def test_pdb_file(test_data_dir):
    """Path to test PDB file."""
    return test_data_dir / "3gd8.pdb"


@pytest.fixture
def test_fixed_pdb_file(test_data_dir):
    """Path to test fixed PDB file."""
    return test_data_dir / "3gd8-fixed.pdb"


@pytest.fixture
def test_trajectory_file(test_data_dir):
    """Path to test trajectory file."""
    return test_data_dir / "traj.dcd"


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def skip_if_no_openmm():
    """Skip tests if OpenMM is not available."""
    return find_spec("openmm") is None
