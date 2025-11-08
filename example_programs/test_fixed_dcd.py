"""Test if the fixed FastDCDReporter produces readable DCD files."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import _example_support
_example_support.ensure_src_on_path()

import openmm
from openmm import app, unit
from openmm.app import PDBFile

from pmarlo.replica_exchange.trajectory import FastDCDReporter


def main():
    """Test the fixed DCD writer."""
    print("\n" + "="*80)
    print("TESTING FIXED FastDCDReporter")
    print("="*80 + "\n")
    
    # Load test system
    data_dir = Path(__file__).parent / "data" / "run-20251108-004416"
    pdb_path = data_dir / "restart" / "3gd8-fixed_run-20251021-122220_run-20251024-201820_run-20251108-004416.pdb"
    
    if not pdb_path.exists():
        print(f"PDB not found: {pdb_path}")
        return 1
    
    print(f"Loading system from: {pdb_path.name}")
    pdb = PDBFile(str(pdb_path))
    
    # Create system
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
    
    print(f"System: {pdb.topology.getNumAtoms()} atoms\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dcd_path = tmpdir / "test.dcd"
        
        # Write with FastDCDReporter
        print(f"Writing DCD with FastDCDReporter...")
        print(f"  Report interval: 100 steps")
        print(f"  Running: 1000 steps\n")
        
        reporter = FastDCDReporter(str(dcd_path), 100)
        simulation.reporters.append(reporter)
        
        simulation.step(1000)
        
        reporter.close()
        simulation.reporters.clear()
        
        file_size = dcd_path.stat().st_size
        print(f"DCD file written: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)\n")
        
        # Try to load with mdtraj
        print("="*80)
        print("LOADING TEST WITH MDTRAJ")
        print("="*80 + "\n")
        
        try:
            import mdtraj as md
            
            print("Attempting to load DCD...")
            traj = md.load(str(dcd_path), top=str(pdb_path))
            
            print(f"\n[OK] SUCCESS!")
            print(f"  Frames loaded: {traj.n_frames}")
            print(f"  Atoms: {traj.n_atoms}")
            print(f"  Time per frame: {traj.timestep} ps")
            print(f"  Total time: {traj.time[-1]:.2f} ps")
            
            print("\n" + "="*80)
            print("[OK] DCD FILE IS VALID AND READABLE!")
            print("="*80 + "\n")
            return 0
            
        except Exception as e:
            print(f"\n[X] FAILED TO LOAD")
            print(f"Error: {e}")
            print(f"Error type: {type(e).__name__}")
            
            import traceback
            traceback.print_exc()
            
            print("\n" + "="*80)
            print("[X] DCD FILE STILL CORRUPTED")
            print("="*80 + "\n")
            return 1


if __name__ == "__main__":
    sys.exit(main())

