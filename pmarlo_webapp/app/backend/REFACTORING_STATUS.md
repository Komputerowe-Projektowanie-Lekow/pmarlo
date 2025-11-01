# Backend Module Refactoring Status

## Completed Refactorings

### ✅ analysis.py (COMPLETED)
**Status:** ✅ Complete (reference implementation)
**Helper Functions:** `analysis_total_pairs()`, `compute_analysis_diag_mass()`
**Mixin:** `AnalysisMixin`

---

### ✅ sampling.py (COMPLETED)

**Changes Applied:**
- [x] All imports moved to top of file
- [x] Imports organized (stdlib → third-party → pmarlo → local)
- [x] Import paths verified and corrected
- [x] Logger created at module level
- [x] Helper function `run_short_sim()` extracted as module-level function
- [x] `SamplingMixin` class created with proper structure
- [x] All methods use `self` parameter in Mixin class
- [x] Type hints added to all function signatures
- [x] Docstrings added to all public functions/classes
- [x] No fallback values used (proper error raising)
- [x] No lazy imports remaining
- [x] File validated - imports successfully

**Module-Level Functions (Importable):**
- `run_short_sim()` - Run a short simulation for testing purposes

**Mixin Methods:**
- `load_run()` - Reconstruct a previous simulation result
- `delete_simulation()` - Delete a simulation run and its files
- `run_sampling()` - Execute a REMD sampling run

---

### ✅ shards.py (COMPLETED)

**Changes Applied:**
- [x] All imports moved to top of file
- [x] Imports organized (stdlib → third-party → pmarlo → local)
- [x] Import paths verified and corrected
  - Added: `import logging`
  - Added: `from pmarlo.data.shard import read_shard`
  - Added: `from pmarlo.utils.path_utils import ensure_directory`
- [x] Logger created at module level
- [x] No module-level helper functions (all methods need self)
- [x] `ShardsMixin` class created with proper structure
- [x] All methods use `self` parameter in Mixin class
- [x] Type hints added to all function signatures
- [x] Docstrings added to all public functions/classes
- [x] No fallback values used (proper error raising)
- [x] No lazy imports remaining
- [x] File validated - imports successfully

**Mixin Methods:**
- `emit_shards()` - Generate shards from a simulation result
- `delete_shard_batch()` - Delete a shard batch and its files
- `discover_shards()` - Discover all shard JSON files
- `shard_summaries()` - Get summaries of all shard batches
- `_reconcile_shard_state()` - Cleanup operation for state consistency

---

### ✅ training.py (COMPLETED)

**Changes Applied:**
- [x] All imports moved to top of file
- [x] Imports organized (stdlib → third-party → pmarlo → local)
- [x] Import paths verified and corrected
  - Added: `import logging`
  - Added: `from pmarlo.utils.path_utils import ensure_directory`
  - Added proper imports for helper functions from utils
- [x] Logger created at module level
- [x] Helper functions extracted as module-level functions
  - `_coerce_hidden_layers()` - Parse hidden layer specification
  - `_coerce_tau_schedule()` - Parse tau schedule
- [x] `TrainingMixin` class created with proper structure
- [x] All methods use `self` parameter in Mixin class
- [x] Type hints added to all function signatures
- [x] Docstrings added to all public functions/classes
- [x] No fallback values used (proper error raising)
- [x] No lazy imports remaining
- [x] File validated - imports successfully

**Module-Level Functions (Importable):**
- `_coerce_hidden_layers()` - Parse hidden layer specification from various formats
- `_coerce_tau_schedule()` - Parse tau schedule from various formats

**Mixin Methods:**
- `train_model()` - Train a DeepTICA model from shards
- `load_model()` - Load a trained model by index
- `delete_model()` - Delete a model and its associated files
- `get_training_progress()` - Read real-time training progress
- `_load_model_from_entry()` - Load model from a state entry
- `_load_build_result_from_path()` - Load a BuildResult from bundle file
- `_export_cv_model()` - Export CV model for OpenMM integration

---

## Modules To Refactor

### 📋 conformations.py
**Status:** Not started
**Estimated complexity:** Medium

### 📋 its.py
**Status:** Not started
**Estimated complexity:** Low

---

## Support Modules (Fixed During Refactoring)

- ✅ `types.py` - Fixed syntax errors, removed incomplete stubs
- ✅ `utils.py` - Fixed missing imports and incomplete function stubs
- ✅ `state.py` - Fixed incomplete class definitions, added PersistentState alias
- ✅ `__init__.py` - Fixed incomplete Backend class, proper mixin imports

---

## Refactoring Pattern Summary

### 1. Import Organization
```python
# Standard library imports (alphabetically)
import logging
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports (alphabetically)
import numpy as np

# pmarlo imports (grouped by submodule)
from pmarlo.api.xxx import function_name
from pmarlo.utils.xxx import function_name

# Local imports (relative)
from .types import TypeName
from .utils import helper_function

logger = logging.getLogger(__name__)
```

### 2. Module-Level Helper Functions
Functions that can be imported by other modules and don't need `self`

### 3. Mixin Class Structure
Methods that need `self` and access to Backend state
