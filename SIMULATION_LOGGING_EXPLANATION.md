# Simulation Logging & Ctrl+C Behavior Explanation

## Problem Summary

When running simulations in the Streamlit app and pressing Ctrl+C, you experienced:
1. A lag where the simulation persisted for some time
2. Multiple progress bars appearing (you saw 2 complete progress bars)
3. Confusion about what the app was doing

## Root Cause Analysis

### The Two-Phase Process

Your replica exchange simulation actually runs in **TWO SEPARATE PHASES**:

1. **PHASE 1: MD SIMULATION** (First progress bar)
   - Runs the actual molecular dynamics simulation
   - Creates trajectory files for each replica
   - This phase CAN be cancelled with Ctrl+C

2. **PHASE 2: DEMULTIPLEXING** (Second progress bar)
   - Extracts frames at the target temperature (300K) from all replica trajectories
   - Creates a single unified trajectory
   - **This phase CANNOT be cancelled with Ctrl+C** - it runs to completion

### Why the Lag?

When you press Ctrl+C:
1. The MD simulation (Phase 1) stops
2. The code immediately proceeds to Phase 2 (demultiplexing)
3. Demultiplexing processes all the trajectories that were already written
4. This causes the "lag" you experienced

The lag isn't the simulation persisting - it's the demultiplexing phase running on the data already generated!

## Solution Implemented

I've added **comprehensive console output using print statements** throughout the simulation pipeline so you can see exactly what's happening in your Streamlit console at each step.

**Why print() instead of logging?**
- Python's `logger` output may not appear in the Streamlit console
- `print(..., flush=True)` guarantees immediate console visibility
- Both `print()` and `logger` are used: print for console, logger for file-based logging

### 1. Startup Banner (`src/pmarlo/api.py`)
```
================================================================================
REPLICA EXCHANGE SIMULATION STARTING
================================================================================
Number of replicas: 3
Temperature ladder: [300.0, 320.0, 340.0]
Total steps: 50000
Output directory: .../replica_exchange
Random seed: None
================================================================================
```

### 2. Phase 1 Logging
```
================================================================================
PHASE 1/2: RUNNING MD SIMULATION
================================================================================
This will run 3 parallel replicas
Each replica will run for 50000 MD steps
Equilibration: 5000 steps
Press Ctrl+C to cancel the simulation
================================================================================

[Equilibration and Production sub-phases are also logged]

================================================================================
PHASE 1/2: MD SIMULATION COMPLETE
================================================================================
Generated 3 replica trajectories
================================================================================
```

### 3. Phase 2 Logging (THE PART THAT DOESN'T STOP WITH CTRL+C)
```
================================================================================
PHASE 2/2: DEMULTIPLEXING TRAJECTORIES
================================================================================
Extracting frames at target temperature (300K)
This creates a single trajectory from replica exchanges
⚠️ This phase cannot be cancelled with Ctrl+C (runs to completion)
================================================================================

[Demux stages and progress are logged]

================================================================================
PHASE 2/2: DEMULTIPLEXING COMPLETE
================================================================================
Total frames written: 1234
Output file: demux_T300K.dcd
================================================================================
```

## How Many Simulations Are Running?

**Answer: Only ONE simulation per button click** (this was fixed earlier)

Your simulation runs **N replicas in parallel** where N = number of temperatures:
- With [300, 320, 340], you run **3 replicas simultaneously**
- Each replica runs independently but exchanges states periodically
- This is normal REMD behavior - it's ONE simulation with multiple parallel replicas

## What You'll See Now

When you run the Streamlit app again, you'll see clear logging like:

```
================================================================================
REPLICA EXCHANGE SIMULATION STARTING
================================================================================
Number of replicas: 3
Temperature ladder: [300.0, 320.0, 340.0]
...

================================================================================
PHASE 1/2: RUNNING MD SIMULATION
================================================================================
...
================================================================================
SIMULATION STAGE 1: EQUILIBRATION (5000 steps)
================================================================================
Running 3 replicas in parallel
Temperature range: 300.0K - 340.0K
================================================================================
[###---------------------------] 10%/100% ETA 0:38

...

================================================================================
PHASE 2/2: DEMULTIPLEXING TRAJECTORIES
================================================================================
⚠️ This phase cannot be cancelled with Ctrl+C (runs to completion)
================================================================================
[###---------------------------] 10%/100% ETA 0:40
```

## Key Takeaways

1. **Two progress bars are normal**: First is simulation, second is demultiplexing
2. **Ctrl+C only stops the simulation phase**: Demux will still run on existing data
3. **The "lag" is demultiplexing**: It's not the simulation persisting
4. **Only ONE simulation runs per click**: Not multiple as you might have thought
5. **Logging is now comprehensive**: You'll always know what phase you're in

## Files Modified

1. `src/pmarlo/api.py` - Main simulation orchestration logging
2. `src/pmarlo/replica_exchange/replica_exchange.py` - Simulation phase logging
3. `src/pmarlo/demultiplexing/demux.py` - Demultiplexing phase logging
4. `changelog.d/20250107-standardize-app-shards.md` - Documentation

## Testing Recommendation

Run a short simulation (e.g., 10,000 steps) and watch the logs:
```bash
poetry run streamlit run example_programs/app_usecase/app/app.py
```

You should see:
- Clear phase boundaries with 80-character separator lines
- What's happening at each stage
- How many replicas are running
- When demux starts (and the warning that Ctrl+C won't stop it)
- Final statistics at completion

## Future Improvements (Optional)

If you want demultiplexing to be cancellable, you would need to:
1. Pass a cancel_token to the demux phase
2. Check the token periodically during frame extraction
3. Handle partial trajectory files gracefully

This wasn't implemented because demux is typically fast compared to simulation.
