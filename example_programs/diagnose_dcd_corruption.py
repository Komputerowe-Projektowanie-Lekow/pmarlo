"""Comprehensive DCD corruption diagnostics using real simulation data.

This program tests DCD file integrity at multiple levels to identify the exact
cause of corruption issues.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import _example_support
_example_support.ensure_src_on_path()


def read_dcd_header_raw(dcd_path: Path) -> dict:
    """Read DCD header using low-level struct operations."""
    print(f"\n{'='*80}")
    print(f"LOW-LEVEL HEADER ANALYSIS: {dcd_path.name}")
    print(f"{'='*80}\n")
    
    results = {}
    
    with open(dcd_path, 'rb') as f:
        file_size = dcd_path.stat().st_size
        print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        results['file_size'] = file_size
        
        try:
            # Block 1: Main header
            block1_size = struct.unpack('<i', f.read(4))[0]
            print(f"\nBlock 1 size: {block1_size} bytes (expected: 84)")
            results['block1_size'] = block1_size
            
            if block1_size != 84:
                print(f"  [X] ERROR: Block 1 size should be 84, got {block1_size}")
                results['block1_valid'] = False
                return results
            results['block1_valid'] = True
            
            # Magic string
            magic = f.read(4)
            print(f"Magic string: {magic} (expected: b'CORD')")
            results['magic'] = magic
            
            if magic != b'CORD':
                print(f"  [X] ERROR: Invalid magic string")
                results['magic_valid'] = False
                return results
            results['magic_valid'] = True
            
            # Frame count (THIS IS CRITICAL)
            frame_count = struct.unpack('<i', f.read(4))[0]
            print(f"Frame count in header: {frame_count}")
            results['frame_count'] = frame_count
            
            # Starting step
            start_step = struct.unpack('<i', f.read(4))[0]
            print(f"Starting step: {start_step}")
            results['start_step'] = start_step
            
            # Step interval
            step_interval = struct.unpack('<i', f.read(4))[0]
            print(f"Step interval: {step_interval}")
            results['step_interval'] = step_interval
            
            # Skip 6 unused ints
            f.read(4 * 6)
            
            # Timestep
            timestep = struct.unpack('<f', f.read(4))[0]
            print(f"Timestep: {timestep:.6f} ps")
            results['timestep'] = timestep
            
            # Skip remaining header fields (10 ints)
            f.read(4 * 10)
            
            # Closing block 1 size
            block1_size_end = struct.unpack('<i', f.read(4))[0]
            print(f"Block 1 closing size: {block1_size_end} (expected: {block1_size})")
            results['block1_size_end'] = block1_size_end
            
            if block1_size != block1_size_end:
                print(f"  [X] ERROR: Block sizes don't match!")
                results['block1_closed_valid'] = False
                return results
            results['block1_closed_valid'] = True
            
            # Block 2: Title
            block2_size = struct.unpack('<i', f.read(4))[0]
            print(f"\nBlock 2 size: {block2_size} bytes")
            results['block2_size'] = block2_size
            
            num_title_lines = struct.unpack('<i', f.read(4))[0]
            title = f.read(block2_size - 4)
            print(f"Title: {title.decode('latin1', errors='ignore')}")
            results['title'] = title
            
            block2_size_end = struct.unpack('<i', f.read(4))[0]
            if block2_size != block2_size_end:
                print(f"  [X] ERROR: Block 2 sizes don't match!")
                results['block2_valid'] = False
                return results
            results['block2_valid'] = True
            
            # Block 3: Number of atoms
            block3_size = struct.unpack('<i', f.read(4))[0]
            print(f"\nBlock 3 size: {block3_size} bytes (expected: 4)")
            
            natoms = struct.unpack('<i', f.read(4))[0]
            print(f"Number of atoms: {natoms}")
            results['natoms'] = natoms
            
            block3_size_end = struct.unpack('<i', f.read(4))[0]
            if block3_size != block3_size_end:
                print(f"  [X] ERROR: Block 3 sizes don't match!")
                results['block3_valid'] = False
                return results
            results['block3_valid'] = True
            
            header_end_pos = f.tell()
            print(f"\nHeader ends at byte: {header_end_pos}")
            results['header_end_pos'] = header_end_pos
            
            # Calculate expected vs actual frames
            # Each frame: cell (56 bytes) + 3 * (8 + natoms*4) for x,y,z
            frame_size = 56 + 3 * (8 + natoms * 4)
            print(f"Calculated frame size: {frame_size} bytes")
            results['frame_size'] = frame_size
            
            expected_data_size = frame_count * frame_size
            actual_data_size = file_size - header_end_pos
            actual_frames = actual_data_size // frame_size
            
            print(f"\nFRAME COUNT ANALYSIS:")
            print(f"  Header claims: {frame_count} frames")
            print(f"  Expected data size: {expected_data_size:,} bytes")
            print(f"  Actual data size: {actual_data_size:,} bytes")
            print(f"  Actual frames that fit: {actual_frames}")
            print(f"  Difference: {actual_frames - frame_count} frames")
            
            results['expected_data_size'] = expected_data_size
            results['actual_data_size'] = actual_data_size
            results['actual_frames'] = actual_frames
            results['frame_count_mismatch'] = actual_frames != frame_count
            
            if actual_frames < frame_count:
                print(f"\n  [X] CORRUPTION DETECTED:")
                print(f"     File is SHORTER than header claims!")
                print(f"     Missing {frame_count - actual_frames} frames")
                print(f"     Missing {expected_data_size - actual_data_size:,} bytes")
                results['corruption_type'] = 'truncated'
                results['is_corrupt'] = True
            elif actual_frames > frame_count:
                print(f"\n  [!] WARNING:")
                print(f"     File has MORE frames than header claims")
                print(f"     Extra {actual_frames - frame_count} frames")
                print(f"     Header frame count may not have been updated")
                results['corruption_type'] = 'stale_header'
                results['is_corrupt'] = True
            else:
                print(f"\n  [OK] Frame count matches!")
                results['is_corrupt'] = False
            
            # Try to read first frame to verify structure
            print(f"\nVERIFYING FIRST FRAME STRUCTURE:")
            try:
                # Cell basis (6 doubles = 48 bytes)
                cell_size = struct.unpack('<i', f.read(4))[0]
                if cell_size != 48:
                    print(f"  [X] ERROR: Cell size should be 48, got {cell_size}")
                    results['first_frame_valid'] = False
                else:
                    f.read(48)  # Skip cell data
                    cell_size_end = struct.unpack('<i', f.read(4))[0]
                    if cell_size != cell_size_end:
                        print(f"  [X] ERROR: Cell block sizes don't match")
                        results['first_frame_valid'] = False
                    else:
                        print(f"  [OK] Cell block OK")
                        results['first_frame_valid'] = True
            except Exception as e:
                print(f"  [X] ERROR reading first frame: {e}")
                results['first_frame_valid'] = False
            
        except Exception as e:
            print(f"\n[X] CRITICAL ERROR reading header: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
            results['is_corrupt'] = True
    
    return results


def test_mdtraj_load(dcd_path: Path, pdb_path: Path) -> dict:
    """Test loading DCD with mdtraj."""
    print(f"\n{'='*80}")
    print(f"MDTRAJ LOADING TEST: {dcd_path.name}")
    print(f"{'='*80}\n")
    
    results = {}
    
    try:
        import mdtraj as md
        
        print(f"Loading with mdtraj...")
        print(f"  DCD: {dcd_path}")
        print(f"  Topology: {pdb_path}")
        
        traj = md.load(str(dcd_path), top=str(pdb_path))
        
        print(f"\n[OK] SUCCESS!")
        print(f"  Frames loaded: {traj.n_frames}")
        print(f"  Atoms: {traj.n_atoms}")
        print(f"  Timestep: {traj.timestep} ps")
        
        results['success'] = True
        results['n_frames'] = traj.n_frames
        results['n_atoms'] = traj.n_atoms
        results['timestep'] = traj.timestep
        
    except Exception as e:
        print(f"\n[X] FAILED TO LOAD WITH MDTRAJ")
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        traceback.print_exc()
        
        results['success'] = False
        results['error'] = str(e)
        results['error_type'] = type(e).__name__
    
    return results


def test_pmarlo_reader(dcd_path: Path, pdb_path: Path) -> dict:
    """Test loading DCD with pmarlo's trajectory reader."""
    print(f"\n{'='*80}")
    print(f"PMARLO READER TEST: {dcd_path.name}")
    print(f"{'='*80}\n")
    
    results = {}
    
    try:
        from pmarlo.io.trajectory_reader import MDTrajReader
        
        print(f"Creating MDTrajReader...")
        reader = MDTrajReader(topology_path=str(pdb_path))
        
        print(f"Probing length...")
        n_frames = reader.probe_length(str(dcd_path))
        
        print(f"\n[OK] SUCCESS!")
        print(f"  Frames detected: {n_frames}")
        
        results['success'] = True
        results['n_frames'] = n_frames
        
    except Exception as e:
        print(f"\n[X] FAILED WITH PMARLO READER")
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        import traceback
        traceback.print_exc()
        
        results['success'] = False
        results['error'] = str(e)
        results['error_type'] = type(e).__name__
    
    return results


def main():
    """Run comprehensive diagnostics on real simulation data."""
    print("\n" + "="*80)
    print("DCD CORRUPTION DIAGNOSTICS")
    print("="*80)
    
    # Find data directory
    data_dir = Path(__file__).parent / "data" / "run-20251108-004416"
    replica_dir = data_dir / "replica_exchange"
    pdb_path = data_dir / "restart" / "3gd8-fixed_run-20251021-122220_run-20251024-201820_run-20251108-004416.pdb"
    
    if not replica_dir.exists():
        print(f"\n[X] ERROR: Data directory not found: {replica_dir}")
        return 1
    
    if not pdb_path.exists():
        print(f"\n[X] ERROR: PDB file not found: {pdb_path}")
        return 1
    
    print(f"\nData directory: {replica_dir}")
    print(f"PDB topology: {pdb_path.name}")
    
    # Find all DCD files
    dcd_files = sorted(replica_dir.glob("replica_*.dcd"))
    print(f"\nFound {len(dcd_files)} DCD files:")
    for dcd in dcd_files:
        size_mb = dcd.stat().st_size / 1024 / 1024
        print(f"  - {dcd.name}: {size_mb:.2f} MB")
    
    if not dcd_files:
        print(f"\n[X] ERROR: No DCD files found")
        return 1
    
    # Test each file
    all_results = {}
    corruption_found = False
    
    for dcd_path in dcd_files:
        print(f"\n\n{'#'*80}")
        print(f"# TESTING: {dcd_path.name}")
        print(f"{'#'*80}")
        
        # Low-level header analysis
        header_results = read_dcd_header_raw(dcd_path)
        
        # MDTraj test
        mdtraj_results = test_mdtraj_load(dcd_path, pdb_path)
        
        # PMARLO reader test
        pmarlo_results = test_pmarlo_reader(dcd_path, pdb_path)
        
        all_results[dcd_path.name] = {
            'header': header_results,
            'mdtraj': mdtraj_results,
            'pmarlo': pmarlo_results
        }
        
        if header_results.get('is_corrupt'):
            corruption_found = True
    
    # Summary
    print(f"\n\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}\n")
    
    for filename, results in all_results.items():
        print(f"\n{filename}:")
        header = results['header']
        mdtraj = results['mdtraj']
        pmarlo = results['pmarlo']
        
        print(f"  Header analysis:")
        if header.get('is_corrupt'):
            print(f"    [X] CORRUPT ({header.get('corruption_type', 'unknown')})")
            print(f"    Header claims: {header.get('frame_count', 'N/A')} frames")
            print(f"    Actual frames: {header.get('actual_frames', 'N/A')} frames")
        else:
            print(f"    [OK] Valid")
            print(f"    Frames: {header.get('frame_count', 'N/A')}")
        
        print(f"  MDTraj:")
        if mdtraj.get('success'):
            print(f"    [OK] Loaded {mdtraj.get('n_frames', 'N/A')} frames")
        else:
            print(f"    [X] Failed: {mdtraj.get('error_type', 'unknown')}")
        
        print(f"  PMARLO Reader:")
        if pmarlo.get('success'):
            print(f"    [OK] Detected {pmarlo.get('n_frames', 'N/A')} frames")
        else:
            print(f"    [X] Failed: {pmarlo.get('error_type', 'unknown')}")
    
    if corruption_found:
        print(f"\n{'='*80}")
        print("[X] CORRUPTION DETECTED IN ONE OR MORE FILES")
        print(f"{'='*80}\n")
        print("PROBABLE CAUSES:")
        print("1. Frame count in header not updated correctly")
        print("2. File not flushed/synced before close")
        print("3. Process killed before final header update")
        print("4. OS caching prevented data from reaching disk")
        return 1
    else:
        print(f"\n{'='*80}")
        print("[OK] ALL FILES APPEAR VALID")
        print(f"{'='*80}\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())

