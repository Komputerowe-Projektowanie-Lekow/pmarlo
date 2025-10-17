# Changelog Entry

## Fixed
- Fixed comprehensive mypy type errors across multiple modules to achieve zero mypy errors for the type tox environment
- Resolved no-redef issues by using distinct names for fallback implementations (EnhancedMSM, prepare_structure)
- Fixed valid-type issues by implementing proper TYPE_CHECKING patterns for conditional type definitions
- Eliminated protocol instantiation errors by using proper type casting and constructor patterns
- Fixed union-attr issues by adding proper None checks before calling methods like `.lower()`
- Resolved arg-type and return-type mismatches in replica_exchange.py by normalizing Path/str types and fixing return statements
- Fixed mixed-type issues in transform/apply.py by adding proper imports, type annotations, and argument handling
- **Removed fallback implementations for TICA and VAMP** to enforce use of standard `deeptime` library implementations only
- **Removed custom MSM fitting fallback (_fit_msm_fallback)** from bridge.py and _msm_utils.py to enforce deeptime-only MSM estimation
- **Eliminated monkeypatching across the entire test suite** to ensure tests use real dependencies and standard library implementations

## Technical Details
- **No-redef fixes**: Renamed stub classes (e.g., `_EnhancedMSMStubClass`) and removed duplicate function definitions to avoid mypy redefinition errors
- **Valid-type fixes**: Used `TYPE_CHECKING` blocks to create type-only aliases (e.g., `EnhancedMSMType`) for conditional class definitions
- **Protocol instantiation fixes**: Applied constructor casting with `cast(Any, Constructor)` pattern when protocols masked real `__init__` signatures
- **Union-attr fixes**: Added explicit None checks and type guards before calling methods on potentially None values
- **Arg-type fixes**: Normalized `str | Path | None` to `str` using assertions and `str()` conversion in ReplicaExchange.from_config
- **Return-type fixes**: Fixed missing return values in run_simulation method by returning appropriate `List[str]` values
- **Transform/apply fixes**: Added missing imports, fixed function signatures, and corrected argument passing for MSM analysis functions
- **Type annotation improvements**: Added proper type annotations to dictionaries and variables to resolve object type inference issues
- **Deeptime dependency enforcement**: 
  - Removed `_manual_tica()` and `_manual_vamp()` fallback implementations from reduction.py
  - Removed `_fit_msm_fallback()` custom MSM fitting from bridge.py and _msm_utils.py
  - All TICA, VAMP, and MSM estimation now exclusively use deeptime library implementations
  - Updated `build_simple_msm()` to directly use deeptime without fallback try/except logic
  - Updated tests to remove monkeypatching and fallback-related test cases
- **Monkeypatching elimination**: Removed all monkeypatching from tests throughout the project:
  - Removed sys.modules mocking for openmm, statsmodels, mdtraj, deeptime in test_experiments_kpi.py
  - Removed scipy.constants fallback mocking in test_thermodynamics.py and enforced scipy as required dependency
  - Replaced implementation detail tests (numpy.cov, sklearn.CCA spying) with behavioral tests in test_diagnostics.py
  - Replaced monkeypatch.chdir with explicit os.chdir/cleanup pattern in test_paths.py
  - Removed deeptime module mocking in test_ck.py
  - Tests now use pytest.importorskip for truly optional dependencies and real implementations for required ones
