# TICA Implementation Investigation Summary

**Date**: October 25, 2025  
**Status**: ✅ VERIFIED - Only deeptime TICA is used, NO fallbacks exist

## Executive Summary

The codebase **exclusively uses the deeptime TICA implementation** with NO silent fallbacks or alternative implementations. This is the correct and desired behavior.

## Key Findings

### 1. Single Source of Truth
- **All TICA usage imports from**: `deeptime.decomposition.TICA`
- **No custom TICA implementation exists**
- **No sklearn-based TICA fallback**
- **No try/except blocks that silently switch implementations**

### 2. TICA Implementation Locations

The codebase uses deeptime's TICA in exactly 3 places:

1. **`src/pmarlo/markov_state_model/reduction.py`** (line 103)
   - Primary reduction API: `tica_reduce()` function
   - Used by: `reduce_features()` unified interface

2. **`src/pmarlo/markov_state_model/_features.py`** (line 182)
   - Used in: `_maybe_apply_tica()` method
   - Part of the MSM pipeline's feature processing

3. **Test files**
   - `tests/unit/markov_state_model/test_reduction.py`
   - Validates TICA matches deeptime's expected behavior

### 3. Bug Fixed

**Issue Identified**: The `tica_reduce()` function was using incorrect deeptime API
- **Before**: `model = tica.fit(X_prep)`
- **After**: `model = tica.fit([X_prep]).fetch_model()`

**Reason**: deeptime's TICA expects:
- `fit()` takes a **list** of trajectory arrays
- `fit()` returns an estimator that needs `.fetch_model()` to get the actual model

This matches the correct usage in `_features.py` line 200.

### 4. Error Handling Policy

✅ **CORRECT BEHAVIOR**: No fallbacks exist
- If deeptime is not installed → `ImportError` raised immediately
- If TICA fails → Exception propagates to caller
- **No silent failures** that would mask issues in unit tests

### 5. Files Examined

```
src/pmarlo/markov_state_model/
├── reduction.py          ✅ FIXED - Now uses correct deeptime API
├── _features.py          ✅ CORRECT - Already using correct API
└── _tica.py              ✅ STUB ONLY - Abstract mixin methods

tests/unit/markov_state_model/
└── test_reduction.py     ✅ PASSING - All 6 tests pass
```

## Test Results

All TICA tests pass with the fix:
```bash
tests/unit/markov_state_model/test_reduction.py::test_tica_reduce_matches_deeptime ✓
tests/unit/markov_state_model/test_reduction.py::test_tica_reduce_nan_handling ✓
tests/unit/markov_state_model/test_reduction.py::test_pca_reduce_matches_sklearn ✓
tests/unit/markov_state_model/test_reduction.py::test_pca_reduce_large_batch_equals_small ✓
tests/unit/markov_state_model/test_reduction.py::test_nan_safe_pca ✓
tests/unit/markov_state_model/test_reduction.py::test_vamp_reduce_nan_handling ✓
```

## Code Changes Made

### File: `src/pmarlo/markov_state_model/reduction.py`

**Function**: `tica_reduce()` (lines 81-112)

```python
# BEFORE (INCORRECT)
def tica_reduce(...):
    from deeptime.decomposition import TICA
    X_prep = _preprocess(X, scale=scale)
    tica = TICA(lagtime=lag, dim=n_components)
    model = tica.fit(X_prep)  # ❌ Wrong: expects list, missing fetch_model()
    transformed = model.transform(X_prep)
    return np.asarray(transformed, dtype=float)

# AFTER (CORRECT)
def tica_reduce(...):
    from deeptime.decomposition import TICA
    X_prep = _preprocess(X, scale=scale)
    tica = TICA(lagtime=lag, dim=n_components)
    model = tica.fit([X_prep]).fetch_model()  # ✅ Correct API usage
    transformed = model.transform(X_prep)
    return np.asarray(transformed, dtype=float)
```

## Recommendations

1. ✅ **Keep current error handling**: NO fallbacks - let exceptions propagate
2. ✅ **Continue using deeptime exclusively**: It's the correct scientific implementation
3. ✅ **Unit tests will catch issues**: Any TICA failures will be visible in tests
4. ✅ **Document deeptime dependency**: Already in requirements, clearly documented

## Conclusion

The codebase follows best practices:
- Single, tested implementation (deeptime)
- No silent fallbacks
- Clear error messages when deeptime is unavailable
- All tests passing with correct API usage

**No further action required** - the investigation confirms the implementation is correct and the identified bug has been fixed.

