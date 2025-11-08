# Single-Temperature MD Strategy

## Overview

When your MSM is **data-starved** (few frames, over-discretized states), prioritize collecting more decorrelated single-temperature data over replica-exchange tweaking.

## The Problem

You have:
- ~9.3k frames at your analysis temperature
- 200 MSM states (over-discretized)
- MSM is starving for transition counts
- REMD shows 100% acceptance (unrealistic, indicates bug)
- Only 3 replicas (marginal REMD benefit)

**Reality check**: Demultiplexed REMD won't fix a data starvation problem. You need **many more frames** at the analysis temperature with good feature coverage.

## The Solution: Single-Temperature MD Ensemble

### Strategy

Run multiple **independent** single-temperature Langevin MD simulations, each with a different random seed:

```python
from pmarlo.api import run_single_temperature_md

# Run 10-20 independent simulations
for seed in range(10):
    trajs, temp = run_single_temperature_md(
        pdb_file="protein.pdb",
        output_dir=f"output/md_run_{seed}",
        temperature=300.0,  # Your analysis temperature
        total_steps=50_000,  # Adjust based on system size
        random_seed=seed,
        target_frames=1000,
    )
```

### Why This Works

**More data where you need it:**
- 100% of compute time at your analysis temperature
- 10 runs × 50k steps = 500k total steps at 300K
- vs. REMD: 3 replicas × 50k steps = only ~17k effective steps at 300K (after demuxing)

**Better statistics:**
- Aim for 50-100 transition pairs per MSM state
- With 200 states, you need 10k-20k transitions minimum
- Single-T ensemble delivers this more efficiently

**Easier to scale:**
- Trivially parallel (run on different machines/GPUs)
- No exchange coordination overhead
- Simple to add more data later (just run more seeds)

**Simpler analysis:**
- No demultiplexing required
- Direct analysis at target temperature
- Clearer interpretation of results

## Implementation

### API Level

```python
from pmarlo.api import run_single_temperature_md, emit_shards_rg_rmsd_windowed

# 1. Run multiple independent MD simulations
trajectory_files = []
for seed in range(10):
    trajs, temp = run_single_temperature_md(
        pdb_file="protein.pdb",
        output_dir=f"output/md_run_{seed}",
        temperature=300.0,
        total_steps=50_000,
        random_seed=seed,
    )
    trajectory_files.extend(trajs)

# 2. Emit shards from all trajectories
shard_paths = emit_shards_rg_rmsd_windowed(
    pdb_file="protein.pdb",
    traj_files=trajectory_files,
    out_dir="output/shards",
    stride=1,
    temperature=300.0,
)

# 3. Build MSM from shards (as usual)
# ...
```

### Webapp Level

In your webapp configuration:

```python
config = SimulationConfig(
    pdb_path=pdb_path,
    temperatures=[300.0],  # Single target temperature
    steps=50_000,
    single_temperature_mode=True,  # Enable single-T mode
    random_seed=seed,
    quick=False,
)

result = backend.run_sampling(config)
```

The webapp backend will automatically:
1. Detect `single_temperature_mode=True`
2. Use `run_single_temperature_md()` instead of `run_replica_exchange()`
3. Return trajectories ready for shard emission
4. Skip demultiplexing (not needed)

### Running Multiple Jobs

**Option 1: Sequential (simple)**
```python
for seed in range(10):
    config.random_seed = seed
    result = backend.run_sampling(config)
```

**Option 2: Parallel (production)**
```bash
# Launch multiple jobs with different seeds
for seed in {0..9}; do
    python run_single_temp.py --seed $seed &
done
wait
```

## Recommended Workflow

### Phase 1: Data Collection (NOW)

1. **Turn off REMD** for initial scouting
2. **Run 10-20 independent single-T jobs**:
   - Each: 50k-100k steps
   - Different seeds: 0, 1, 2, ..., 19
   - Save at stride to get 50-100 frames per state
3. **Target**: ≥10k-20k total frames at 300K

### Phase 2: MSM Building

1. **Emit shards** from all trajectories
2. **Build MSM** with reduced state count (e.g., 50 states instead of 200)
3. **Check transition counts**: aim for 50-100 per state
4. **If still data-starved**: run more single-T jobs

### Phase 3: Enhanced Sampling (LATER)

Only after you have adequate statistics at the target temperature:

1. **Optimize REMD** if enhanced sampling is needed
2. **Fix acceptance logging** (100% is a bug)
3. **Use 8-16 replicas** (not 3) with 20-40% acceptance
4. **Combine** REMD and single-T data for best results

## Parameter Guidelines

### System Size

- **Small peptide** (10-20 residues): 50k steps/run
- **Small protein** (50-100 residues): 100k steps/run  
- **Large protein** (>100 residues): 200k+ steps/run

### Number of Runs

- **Initial scout**: 5-10 runs
- **Production**: 10-20 runs
- **Deep sampling**: 20-50 runs

### Stride Selection

```python
# Rule of thumb: save enough frames for good statistics
frames_needed = n_states * 100  # 100 transitions per state
stride = max(1, production_steps // frames_needed)
```

### MSM State Count

**Don't over-discretize!**

- **Initial**: 30-50 states
- **After validation**: 50-100 states
- **Danger zone**: >150 states (needs massive data)

## Comparison: REMD vs Single-T

| Aspect | REMD (3 replicas) | Single-T Ensemble (10 runs) |
|--------|-------------------|----------------------------|
| Compute at target T | 33% (1/3 replicas) | 100% (all runs) |
| Total frames at 300K | ~3k (after demux) | ~10k (direct) |
| Parallelization | Coupled (exchanges) | Independent (trivial) |
| Demultiplexing | Required | Not needed |
| Acceptance tuning | Complex | N/A |
| Adding more data | Re-run everything | Just add more seeds |
| Analysis complexity | High | Low |
| MSM statistics | Weak (data-starved) | Strong (well-fed) |

## Troubleshooting

**Q: My MSM is still undersampled**
- A: Run more independent simulations (increase n_runs)
- A: Reduce MSM state count (fewer states = more counts per state)
- A: Check tau (lag time) - might be too short

**Q: Should I combine REMD and single-T data?**
- A: Yes, but only after fixing REMD acceptance issues
- A: Single-T data provides baseline, REMD adds enhanced sampling

**Q: How do I know when I have enough data?**
- A: Check transition matrix connectivity
- A: Verify 50-100 counts per state
- A: Validate with Chapman-Kolmogorov test

## References

See `example_programs/single_temp_strategy.py` for a complete working example.

## Summary

**Current bottleneck**: Tiny dataset at target temperature + over-discretized MSM

**Solution**: More decorrelated single-temperature data

**Implementation**: Set `single_temperature_mode=True` in webapp config

**Next steps**:
1. Run 10-20 single-T simulations (different seeds)
2. Emit shards and build MSM
3. Validate statistics before optimizing REMD

