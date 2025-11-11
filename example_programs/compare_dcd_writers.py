"""Compare DCD files written by OpenMM's DCDReporter vs our FastDCDReporter.

This will help identify what's different in the header format.
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

import _example_support
_example_support.ensure_src_on_path()

import numpy as np
import openmm
from openmm import app, unit
from openmm.app import PDBFile

from pmarlo.replica_exchange.trajectory import FastDCDReporter


def read_dcd_header_bytes(dcd_path: Path, num_bytes: int = 200) -> bytes:
    """Read the first N bytes of a DCD file."""
    with open(dcd_path, 'rb') as f:
        return f.read(num_bytes)


def parse_header_fields(data: bytes) -> dict:
    """Parse DCD header fields."""
    fields = {}

    # Block 1
    fields['block1_size'] = struct.unpack('<i', data[0:4])[0]
    fields['magic'] = data[4:8]
    fields['nframes'] = struct.unpack('<i', data[8:12])[0]
    fields['start_step'] = struct.unpack('<i', data[12:16])[0]
    fields['step_interval'] = struct.unpack('<i', data[16:20])[0]

    # Skip 6 ints (24 bytes)
    fields['timestep'] = struct.unpack('<f', data[44:48])[0]

    # Next 10 ints starting at byte 48
    int_fields = struct.unpack('<10i', data[48:88])
    fields['int_field_0'] = int_fields[0]
    fields['int_field_8'] = int_fields[8]
    fields['int_field_9'] = int_fields[9]  # This is the problematic one (24 in ours)

    return fields


def create_test_system():
    """Create a simple test system for comparison."""
    # Find a PDB file
    data_dir = Path(__file__).parent / "data" / "run-20251108-004416"
    pdb_path = data_dir / "restart" / "3gd8-fixed_run-20251021-122220_run-20251024-201820_run-20251108-004416.pdb"

    if not pdb_path.exists():
        print(f"PDB not found: {pdb_path}")
        # Try finding any PDB in tests
        test_assets = Path(__file__).parent.parent / "tests" / "_assets"
        pdb_files = list(test_assets.glob("*.pdb"))
        if pdb_files:
            pdb_path = pdb_files[0]
            print(f"Using: {pdb_path}")
        else:
            print("No PDB file available for testing")
            return None, None

    pdb = PDBFile(str(pdb_path))

    # Create a simple system
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds
    )

    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtosecond
    )

    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    return simulation, pdb


def main():
    """Compare DCD headers between OpenMM and our implementation."""
    print("\n" + "="*80)
    print("DCD WRITER COMPARISON: OpenMM vs FastDCDReporter")
    print("="*80 + "\n")

    # Create test system
    print("Creating test system...")
    simulation, pdb = create_test_system()

    if simulation is None:
        print("Failed to create test system")
        return 1

    print(f"System: {simulation.topology.getNumAtoms()} atoms")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write with OpenMM's DCDReporter
        print("\n[1] Writing with OpenMM's DCDReporter...")
        openmm_dcd = tmpdir / "openmm.dcd"
        openmm_reporter = app.DCDReporter(str(openmm_dcd), 100)
        simulation.reporters.append(openmm_reporter)

        # Run a few steps
        simulation.step(500)

        # Close reporter
        del openmm_reporter
        simulation.reporters.clear()

        print(f"    Written: {openmm_dcd.stat().st_size} bytes")

        # Write with our FastDCDReporter
        print("\n[2] Writing with FastDCDReporter...")
        fast_dcd = tmpdir / "fast.dcd"
        fast_reporter = FastDCDReporter(str(fast_dcd), 100)
        simulation.reporters.append(fast_reporter)

        # Reset and run again
        simulation.context.setPositions(pdb.positions)
        simulation.step(500)

        # Close reporter
        fast_reporter.close()
        simulation.reporters.clear()

        print(f"    Written: {fast_dcd.stat().st_size} bytes")

        # Read and compare headers
        print("\n" + "="*80)
        print("HEADER COMPARISON")
        print("="*80 + "\n")

        openmm_header = read_dcd_header_bytes(openmm_dcd)
        fast_header = read_dcd_header_bytes(fast_dcd)

        openmm_fields = parse_header_fields(openmm_header)
        fast_fields = parse_header_fields(fast_header)

        print("Field                  OpenMM          FastDCD         Match")
        print("-"*70)

        for field in sorted(openmm_fields.keys()):
            openmm_val = openmm_fields[field]
            fast_val = fast_fields[field]

            if isinstance(openmm_val, bytes):
                match = "YES" if openmm_val == fast_val else "NO"
                print(f"{field:20s}   {openmm_val!r:15s} {fast_val!r:15s} {match}")
            elif isinstance(openmm_val, float):
                match = "YES" if abs(openmm_val - fast_val) < 1e-6 else "NO"
                print(f"{field:20s}   {openmm_val:15.6f} {fast_val:15.6f} {match}")
            else:
                match = "YES" if openmm_val == fast_val else "NO"
                print(f"{field:20s}   {openmm_val:15d} {fast_val:15d} {match}")

        # Show hex dump of differences
        print("\n" + "="*80)
        print("HEX DUMP OF FIRST 100 BYTES")
        print("="*80 + "\n")

        print("OpenMM DCDReporter:")
        for i in range(0, min(100, len(openmm_header)), 16):
            hex_str = ' '.join(f'{b:02x}' for b in openmm_header[i:i+16])
            print(f"  {i:04x}: {hex_str}")

        print("\nFastDCDReporter:")
        for i in range(0, min(100, len(fast_header)), 16):
            hex_str = ' '.join(f'{b:02x}' for b in fast_header[i:i+16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in fast_header[i:i+16])
            print(f"  {i:04x}: {hex_str}  |{ascii_str}|")

        print("\nDifferences:")
        differences = []
        for i in range(min(len(openmm_header), len(fast_header))):
            if openmm_header[i] != fast_header[i]:
                differences.append((i, openmm_header[i], fast_header[i]))

        if differences:
            print(f"Found {len(differences)} byte differences:")
            for offset, openmm_byte, fast_byte in differences[:20]:  # Show first 20
                print(f"  Byte {offset}: OpenMM={openmm_byte:02x} FastDCD={fast_byte:02x}")
        else:
            print("Headers are identical!")

        # Try to load both with mdtraj
        print("\n" + "="*80)
        print("MDTRAJ LOADING TEST")
        print("="*80 + "\n")

        try:
            import mdtraj as md

            print("[1] Loading OpenMM DCD...")
            try:
                traj1 = md.load(str(openmm_dcd), top=str(pdb))
                print(f"    SUCCESS: {traj1.n_frames} frames")
            except Exception as e:
                print(f"    FAILED: {e}")

            print("\n[2] Loading FastDCD...")
            try:
                traj2 = md.load(str(fast_dcd), top=str(pdb))
                print(f"    SUCCESS: {traj2.n_frames} frames")
            except Exception as e:
                print(f"    FAILED: {e}")

        except ImportError:
            print("mdtraj not available for testing")

    return 0


if __name__ == "__main__":
    sys.exit(main())
