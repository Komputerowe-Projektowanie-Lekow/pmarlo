"""Regression tests for DCD format compliance.

These tests ensure that FastDCDReporter writes DCD files that are fully
compatible with VMD/MDTraj/CHARMM standards.
"""

from __future__ import annotations

import gc
import struct
import tempfile
from pathlib import Path

import numpy as np
import openmm
import pytest
from openmm import app, unit

from pmarlo.replica_exchange.trajectory import FastDCDReporter


@pytest.fixture
def test_system():
    """Create a simple test system."""
    from openmm.app import ForceField, PDBFile

    # Use test data
    test_pdb = Path(__file__).parent.parent.parent / "_assets" / "3gd8-fixed.pdb"
    if not test_pdb.exists():
        pytest.skip(f"Test PDB not found: {test_pdb}")

    pdb = PDBFile(str(test_pdb))
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
    )

    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond
    )

    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    return simulation, pdb, test_pdb


def read_dcd_header(dcd_path: Path) -> dict:
    """Read and validate DCD header format."""
    with open(dcd_path, "rb") as f:
        # Block 1
        block1_size = struct.unpack("<i", f.read(4))[0]
        assert block1_size == 84, f"Block 1 size should be 84, got {block1_size}"

        magic = f.read(4)
        assert magic == b"CORD", f"Magic should be CORD, got {magic}"

        nframes = struct.unpack("<i", f.read(4))[0]
        start_step = struct.unpack("<i", f.read(4))[0]
        step_interval = struct.unpack("<i", f.read(4))[0]

        # Skip 6 unused ints
        f.read(24)

        timestep = struct.unpack("<f", f.read(4))[0]

        # Skip 10 more ints
        f.read(40)

        block1_end = struct.unpack("<i", f.read(4))[0]
        assert block1_end == 84, "Block 1 closing size mismatch"

        # Block 2 (title)
        block2_size = struct.unpack("<i", f.read(4))[0]
        num_title_lines = struct.unpack("<i", f.read(4))[0]
        title_data = f.read(block2_size - 4)
        block2_end = struct.unpack("<i", f.read(4))[0]
        assert block2_end == block2_size, "Block 2 closing size mismatch"

        # Block 3 (natoms)
        block3_size = struct.unpack("<i", f.read(4))[0]
        assert block3_size == 4, "Block 3 size should be 4"
        natoms = struct.unpack("<i", f.read(4))[0]
        block3_end = struct.unpack("<i", f.read(4))[0]
        assert block3_end == 4, "Block 3 closing size mismatch"

        return {
            "nframes": nframes,
            "start_step": start_step,
            "step_interval": step_interval,
            "timestep": timestep,
            "num_title_lines": num_title_lines,
            "title_data": title_data,
            "natoms": natoms,
        }


def test_dcd_header_format_compliance(test_system):
    """Test that DCD header follows CHARMM format specification."""
    simulation, pdb, pdb_path = test_system

    with tempfile.TemporaryDirectory() as tmpdir:
        dcd_path = Path(tmpdir) / "test.dcd"

        # Write DCD with reportInterval=100
        reporter = FastDCDReporter(str(dcd_path), reportInterval=100)
        simulation.reporters.append(reporter)

        # Run simulation
        simulation.step(500)  # Should write 5 frames

        reporter.close()

        # Read and validate header
        header = read_dcd_header(dcd_path)

        # Check critical fields
        assert header["nframes"] == 5, f"Expected 5 frames, got {header['nframes']}"
        assert (
            header["start_step"] == 100
        ), f"start_step should equal reportInterval (100), got {header['start_step']}"
        assert (
            header["step_interval"] == 100
        ), f"step_interval should be 100, got {header['step_interval']}"

        # Timestep should be integration timestep (2 fs = 0.002 ps)
        assert (
            abs(header["timestep"] - 0.002) < 1e-6
        ), f"timestep should be 0.002 ps, got {header['timestep']}"

        # Title should have 2 lines of 80 characters each
        assert (
            header["num_title_lines"] == 2
        ), f"Should have 2 title lines, got {header['num_title_lines']}"
        assert (
            len(header["title_data"]) == 160
        ), f"Title should be 160 bytes (2x80), got {len(header['title_data'])}"


def test_dcd_readable_by_mdtraj(test_system):
    """Test that DCD files can be read by mdtraj."""
    simulation, pdb, pdb_path = test_system

    with tempfile.TemporaryDirectory() as tmpdir:
        dcd_path = Path(tmpdir) / "test.dcd"

        reporter = FastDCDReporter(str(dcd_path), reportInterval=50)
        simulation.reporters.append(reporter)

        simulation.step(1000)  # Should write 20 frames

        reporter.close()

        # Load with mdtraj
        import mdtraj as md

        traj = md.load(str(dcd_path), top=str(pdb_path))

        assert traj.n_frames == 20, f"Expected 20 frames, got {traj.n_frames}"
        assert traj.n_atoms == pdb.topology.getNumAtoms()


def test_dcd_frame_count_consistency(test_system):
    """Test that frame count in header matches actual frames in file."""
    simulation, pdb, pdb_path = test_system

    with tempfile.TemporaryDirectory() as tmpdir:
        dcd_path = Path(tmpdir) / "test.dcd"

        reporter = FastDCDReporter(str(dcd_path), reportInterval=100)
        simulation.reporters.append(reporter)

        simulation.step(1500)  # Should write 15 frames

        reporter.close()

        # Read header
        header = read_dcd_header(dcd_path)

        # Calculate expected file size
        natoms = header["natoms"]
        nframes_header = header["nframes"]

        # Each frame: cell (56 bytes) + 3 * (8 + natoms*4) for x,y,z
        frame_size = 56 + 3 * (8 + natoms * 4)

        # Header size is at byte 149 for this format (approximate)
        file_size = dcd_path.stat().st_size

        # Data size should accommodate exactly nframes_header frames
        # (allowing for header which we calculate from the format)
        header_size = 4 + 84 + 4 + 4 + 164 + 4 + 4 + 4 + 4  # Approximate header
        data_size = file_size - header_size
        actual_frames = data_size // frame_size

        # Frame count should be consistent
        assert (
            nframes_header == 15
        ), f"Header should claim 15 frames, got {nframes_header}"
        assert (
            actual_frames >= 15
        ), f"File should contain at least 15 frames, got {actual_frames}"


def test_dcd_update_frequency(test_system):
    """Test that header frame count is updated periodically."""
    simulation, pdb, pdb_path = test_system

    with tempfile.TemporaryDirectory() as tmpdir:
        dcd_path = Path(tmpdir) / "test.dcd"

        # Use update_frequency=5 for testing
        reporter = FastDCDReporter(str(dcd_path), reportInterval=10, update_frequency=5)
        simulation.reporters.append(reporter)

        # Write some frames
        simulation.step(100)  # Should write 10 frames

        # Manually flush and check frame count
        reporter.flush()

        # Read current frame count from file
        with open(dcd_path, "rb") as f:
            f.seek(8)  # Frame count is at byte 8
            frame_count = struct.unpack("<i", f.read(4))[0]

        assert (
            frame_count == 10
        ), f"Frame count should be 10 after flush, got {frame_count}"

        reporter.close()


def test_dcd_multiple_reporters_different_intervals(test_system):
    """Test that different reportIntervals produce valid files."""
    simulation, pdb, pdb_path = test_system

    intervals = [10, 50, 100, 200]

    for interval in intervals:
        with tempfile.TemporaryDirectory() as tmpdir:
            dcd_path = Path(tmpdir) / f"test_{interval}.dcd"

            reporter = FastDCDReporter(str(dcd_path), reportInterval=interval)
            simulation.reporters.append(reporter)

            # Reset simulation
            simulation.context.setPositions(pdb.positions)
            simulation.step(1000)

            reporter.close()
            simulation.reporters.clear()

            # Verify header
            header = read_dcd_header(dcd_path)

            expected_frames = 1000 // interval
            assert (
                header["nframes"] == expected_frames
            ), f"Interval {interval}: expected {expected_frames} frames, got {header['nframes']}"
            assert (
                header["start_step"] == interval
            ), f"Interval {interval}: start_step should be {interval}, got {header['start_step']}"
            assert (
                header["step_interval"] == interval
            ), f"Interval {interval}: step_interval should be {interval}, got {header['step_interval']}"


def test_dcd_vs_openmm_dcdreporter_compatibility(test_system):
    """Test that our DCD files are compatible with OpenMM's DCDReporter output."""
    simulation, pdb, pdb_path = test_system

    with tempfile.TemporaryDirectory() as tmpdir:
        fast_dcd = Path(tmpdir) / "fast.dcd"
        openmm_dcd = Path(tmpdir) / "openmm.dcd"

        # Write with FastDCDReporter
        fast_reporter = FastDCDReporter(str(fast_dcd), 100)
        simulation.reporters.append(fast_reporter)
        simulation.step(500)
        fast_reporter.close()
        simulation.reporters.clear()

        # Reset and write with OpenMM's DCDReporter
        simulation.context.setPositions(pdb.positions)
        openmm_reporter = app.DCDReporter(str(openmm_dcd), 100)
        simulation.reporters.append(openmm_reporter)
        simulation.step(500)
        del openmm_reporter
        simulation.reporters.clear()

        # Both should be readable by mdtraj
        import mdtraj as md

        traj_fast = md.load(str(fast_dcd), top=str(pdb_path))
        traj_openmm = md.load(str(openmm_dcd), top=str(pdb_path))

        # Should have same number of frames
        assert (
            traj_fast.n_frames == traj_openmm.n_frames
        ), f"Frame count mismatch: Fast={traj_fast.n_frames}, OpenMM={traj_openmm.n_frames}"

        # Should have same number of atoms
        assert traj_fast.n_atoms == traj_openmm.n_atoms


def test_dcd_crash_recovery_with_frequent_updates(test_system):
    """Test that files with frequent header updates remain valid."""
    simulation, pdb, pdb_path = test_system

    with tempfile.TemporaryDirectory() as tmpdir:
        dcd_path = Path(tmpdir) / "test.dcd"

        # Use very frequent updates (every frame)
        reporter = FastDCDReporter(str(dcd_path), reportInterval=10, update_frequency=1)
        simulation.reporters.append(reporter)

        simulation.step(500)  # 50 frames

        # Simulate a crash by not calling close()
        # Just remove the reporter
        simulation.reporters.clear()
        del reporter
        gc.collect()

        # File should still be readable because header was updated frequently
        import mdtraj as md

        traj = md.load(str(dcd_path), top=str(pdb_path))

        # Should have most frames (might miss the last one if not flushed)
        assert (
            traj.n_frames >= 48
        ), f"Expected at least 48 frames after 'crash', got {traj.n_frames}"
