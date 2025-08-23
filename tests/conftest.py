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


@pytest.fixture
def damaged_pdb_file(tmp_path):
    """Create a deliberately damaged PDB file."""
    path = tmp_path / "damaged.pdb"
    path.write_text("ATOM      1  N   ALA A   1\nEND\n")
    return path


@pytest.fixture
def nan_pdb_file(tmp_path):
    """Create a PDB file containing NaN coordinates."""
    from openmm import Vec3, unit
    from openmm.app import PDBFile, Topology, element

    path = tmp_path / "nan.pdb"

    top = Topology()
    chain = top.addChain("A")
    res = top.addResidue("ALA", chain)
    top.addAtom("N", element.get_by_symbol("N"), res)
    top.addAtom("CA", element.get_by_symbol("C"), res)
    positions = unit.Quantity(
        [Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0)],
        unit.nanometer,
    )
    with open(path, "w") as handle:
        PDBFile.writeFile(top, positions, handle)

    lines = path.read_text().splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("ATOM") and line[12:16].strip() == "N":
            line = line[:30] + f"{'NaN':>8}" + line[38:]
        elif line.startswith("ATOM") and line[12:16].strip() == "CA":
            line = line[:38] + f"{'NaN':>8}" + line[46:]
        new_lines.append(line)
    path.write_text("\n".join(new_lines) + "\n")
    return path
