# TPT Conformations Analysis - Streamlit App Integration

## ✅ Integration Complete

The TPT (Transition Path Theory) conformations analysis module has been successfully integrated into the PMARLO Streamlit app at `example_programs/app_usecase/app/app.py`.

## 🎯 What Was Done

### 1. **Fixed Import Errors** ✅
- **Problem**: `find_conformations_tpt_real.py` was importing from `pmarlo.api`, which triggered module-level deeptime imports and caused `ModuleNotFoundError`.
- **Solution**: Replaced `pmarlo.api` imports with direct imports from specific modules:
  - `pmarlo.markov_state_model.bridge.build_simple_msm`
  - `pmarlo.markov_state_model.clustering.cluster_microstates`
  - `pmarlo.markov_state_model.reduction.reduce_features`
- **Result**: `find_conformations_tpt_real.py` now runs without import errors.

### 2. **Backend Functions Added** ✅
Added to `example_programs/app_usecase/app/backend.py`:

#### New Data Classes:
```python
@dataclass
class ConformationsConfig:
    """Configuration for TPT conformations analysis."""
    lag: int = 10
    n_clusters: int = 30
    cluster_mode: str = "kmeans"
    cluster_seed: Optional[int] = 42
    kmeans_kwargs: Dict[str, Any] = field(default_factory=lambda: {"n_init": 50})
    n_components: int = 3
    n_metastable: int = 4
    temperature: float = 300.0
    auto_detect_states: bool = True
    source_states: Optional[List[int]] = None
    sink_states: Optional[List[int]] = None
    n_paths: int = 10
    pathway_fraction: float = 0.99
    compute_kis: bool = True
    k_slow: int = 3
    uncertainty_analysis: bool = True
    bootstrap_samples: int = 50
    n_representatives: int = 5

@dataclass
class ConformationsResult:
    """Result from TPT conformations analysis."""
    output_dir: Path
    tpt_summary: Dict[str, Any]
    metastable_states: Dict[str, Any]
    transition_states: List[Dict[str, Any]]
    pathways: List[List[int]]
    representative_pdbs: List[Path]
    plots: Dict[str, Path]
    created_at: str
    config: ConformationsConfig
    error: Optional[str] = None
```

#### New Method:
```python
def run_conformations_analysis(
    self,
    shard_jsons: Sequence[Path],
    config: ConformationsConfig,
) -> ConformationsResult:
    """Run TPT conformations analysis on shards."""
```

This method:
1. Loads shards and extracts features (Rg, RMSD)
2. Performs TICA dimensionality reduction
3. Clusters into microstates using the configured KMeans-based method
4. Builds MSM with deeptime
5. Runs TPT analysis via `find_conformations()`
   - Uncertainty quantification settings (`uncertainty_analysis`, `bootstrap_samples`) now come from `ConformationsConfig` and are forwarded directly, so the workflow either performs or skips bootstrap analysis explicitly with no silent fallbacks.
6. Generates visualizations (committors, flux networks, pathways)
7. Saves representative PDB structures
8. Exports summary JSON

### 3. **Streamlit App UI Integration** ✅
Added to `example_programs/app_usecase/app/app.py` in the **Analysis tab**:

#### New Section: "TPT Conformations Analysis"
Located after the "Build MSM/FES" section, it includes:

**Configuration Panel** (in collapsible expander):
- Lag (steps)
- N microstates (clusters)
- TICA components
- N metastable states
- Temperature (K)
- Max pathways
- Auto-detect source/sink states (checkbox)
- Compute Kinetic Importance Score (checkbox)

**"Run Conformations Analysis" Button**:
- Executes the full TPT workflow on selected shards
- Shows progress spinner during analysis
- Displays results when complete

**Results Display**:
1. **TPT Metrics** (4 columns):
   - Rate
   - MFPT (Mean First Passage Time)
   - Total Flux
   - Number of Pathways

2. **Metastable States Table**:
   - State ID
   - Population
   - Number of microstates
   - Representative PDB filename

3. **Transition States Table**:
   - State Index
   - Committor probability
   - Representative PDB filename

4. **Visualizations** (2-column grid):
   - TPT summary plot
   - Committor plots (forward/backward)
   - Flux network (gross/net flux)
   - Pathway decomposition

5. **Output Information**:
   - Output directory path
   - Number of representative PDB files saved

### 4. **Visualization Functions** ✅
Already implemented in `src/pmarlo/conformations/visualizations.py`:
- `plot_tpt_summary()` - Complete TPT overview
- `plot_committors()` - Forward/backward committor heatmaps
- `plot_flux_network()` - Network graph of state transitions
- `plot_pathways()` - Top pathways visualization

### 5. **Example Programs** ✅

#### `example_programs/find_conformations_tpt.py`
Demo with test data.

#### `example_programs/find_conformations_tpt_real.py`
Analysis on real shards with CLI selection:
```bash
# Use specific shards
python find_conformations_tpt_real.py --shards 0,1,2,3

# Use all shards in a dataset
python find_conformations_tpt_real.py --dataset mixed_ladders_shards
```

## 🚀 How to Use in the App

### Step 1: Launch the Streamlit App
```bash
cd example_programs/app_usecase/app
poetry run streamlit run app.py
```

### Step 2: Navigate to the Analysis Tab
Click on the **"Analysis"** tab in the app.

### Step 3: Select Shards
1. In the "Shard groups for analysis" multiselect, choose which shard batches to analyze
2. The app will display how many shard files are selected

### Step 4: Configure Conformations Analysis
1. Scroll down to the **"TPT Conformations Analysis"** section
2. Click on **"Configure Conformations Analysis"** expander to adjust parameters:
   - **Lag**: Lag time for MSM building (default: 10)
   - **N microstates**: Number of clusters (default: 30)
   - **Clustering method**: Choice of KMeans, MiniBatchKMeans, or auto-selection
   - **Clustering seed**: Deterministic seed for clustering (-1 disables)
   - **K-means n_init**: Restarts forwarded to the clustering estimator
   - **TICA components**: Dimensionality reduction (default: 3)
   - **N metastable states**: Number of coarse-grained states (default: 4)
   - **Temperature**: Reference temperature in Kelvin (default: 300)
   - **Max pathways**: Maximum number of pathways to extract (default: 10)
   - **Auto-detect source/sink states**: Automatically identify source and sink states from FES (default: True)
   - **Compute Kinetic Importance Score**: Calculate KIS for states (default: True)

### Step 5: Run Analysis
1. Click the **"Run Conformations Analysis"** button (primary blue button)
2. Wait for the analysis to complete (progress spinner will show)
3. Results will display automatically when finished

### Step 6: Explore Results
The app will show:
- **TPT Metrics**: Rate, MFPT, Total Flux, N Pathways
- **Metastable States**: Table with populations and representative PDBs
- **Transition States**: Table with committor probabilities and PDBs
- **Visualizations**: Committors, flux networks, pathways
- **Output Directory**: Location of saved files

## 📁 Output Files

All outputs are saved to: `app_outputs/bundles/conformations-<timestamp>/`

### Files Created:
1. **Representative PDB Structures**:
   - `metastable_1.pdb`, `metastable_2.pdb`, ... (metastable states)
   - `transition_1.pdb`, `transition_2.pdb`, ... (transition states)
   - `stable_1.pdb`, `stable_2.pdb`, ... (stable states)

2. **Visualizations**:
   - `tpt_summary.png` - Complete TPT overview
   - `committors.png` - Forward/backward committor plots
   - `flux_network.png` - State transition network
   - `pathways.png` - Top reactive pathways

3. **Summary JSON**:
   - `conformations_summary.json` - All analysis results in JSON format

## 🔧 Technical Details

### Direct Deeptime Integration (NO FALLBACKS)
The implementation strictly follows the `deeptime` documentation:
- Uses `MarkovStateModel.reactive_flux(source, sink)` directly
- Uses `ReactiveFlux.pathways(fraction, maxiter)` for pathway decomposition
- Uses `ReactiveFlux.coarse_grain(sets)` for coarse-graining
- **NO try-except fallbacks** - raises `ImportError` immediately if deeptime is missing

### Error Handling
- Clear error messages displayed in the app UI
- Failed analyses return `ConformationsResult` with `error` field populated
- Tracebacks logged for debugging

### Data Flow
```
Shards → Features → TICA → Clustering → MSM → TPT → Conformations → PDBs + Plots
```

1. **Load shards** from JSON files
2. **Extract features**: Rg, RMSD_ref
3. **Reduce dimensionality**: TICA with configurable lag and components
4. **Cluster**: MiniBatchKMeans with configurable n_clusters
5. **Build MSM**: Transition matrix T and stationary distribution π
6. **Run TPT**: Auto-detect or use provided source/sink states
7. **Extract conformations**: Metastable, transition, and pathway states
8. **Save outputs**: PDB structures, plots, JSON summary

## ✅ Tests Passing

All conformations unit tests pass:
```bash
$ poetry run pytest tests/unit/conformations/ -v
============================= test session starts =============================
tests\unit\conformations\test_tpt_analysis.py .......                    [ 41%]
tests\unit\conformations\test_kinetic_importance.py ....                 [ 64%]
tests\unit\conformations\test_state_detection.py ......                  [100%]
======================= 17 passed, 2 warnings in 3.44s =======================
```

## 📝 Changelog Updated

All changes documented in `changelog.d/20250309_deeptime_clustering.md`:
- Streamlit app integration details
- Backend functions
- UI features
- Output formats
- Import fixes

## 🎉 Ready to Use!

The TPT conformations analysis is now fully integrated and ready to use in the PMARLO Streamlit app. You can:
1. Select shards from your experiments
2. Configure analysis parameters
3. Click a button to run the analysis
4. View results in the app
5. Download representative PDB structures
6. Use the visualizations for presentations

**No command-line required** - everything is accessible through the web UI! 🚀

