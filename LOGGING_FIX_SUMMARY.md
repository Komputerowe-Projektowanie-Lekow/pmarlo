# Logging Configuration Fix Summary

**Date:** October 25, 2025

## Problem Statement

The Streamlit app was experiencing two logging issues:
1. **Multiple log files being created** - Each Streamlit rerun was creating a new log file
2. **DEBUG messages not appearing in logs** - Messages like "Shard X: SKIPPED - empty trajectory" were not being captured

## Root Causes Identified

### Issue 1: Multiple Log Files
- Streamlit calls `main()` on every user interaction (reruns)
- `_configure_file_logging()` was being called on every rerun
- Each call created a new timestamped log file and added new handlers

### Issue 2: Missing DEBUG Messages
- Some "SKIPPED" messages in `src/pmarlo/reporting/plots.py` were using `print()` statements instead of `logger.debug()`
- These print statements bypass the logging system entirely and go directly to console

## Changes Made

### 1. Fixed Multiple Log Files (`example_programs/app_usecase/app/app.py`)

**Location:** `_configure_file_logging()` function (lines 92-158)

**Change:** Added singleton pattern with handler check:

```python
def _configure_file_logging() -> None:
    # Singleton check: Only configure logging once
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        # Logging already configured, skip
        return
    
    # ... rest of configuration only runs once
```

**Behavior:**
- ✅ First call: Full logging setup runs, creates ONE log file
- ✅ Subsequent calls: Returns immediately, no new files or handlers
- ✅ No silent fallbacks: Raises OSError if directory/file creation fails

### 2. Fixed Missing DEBUG Messages (`src/pmarlo/reporting/plots.py`)

**Location:** `plot_sampling_validation()` function (lines 269-306)

**Changes:** Replaced all `print()` statements with `logger.debug()`:

```python
# BEFORE:
print(f"[DEBUG] Shard {i}: SKIPPED - empty trajectory")

# AFTER:
logger.debug(f"Shard {i}: SKIPPED - empty trajectory")
```

**Files Modified:**
- Lines 271-306 in `plot_sampling_validation()` function
- All debug print statements converted to proper logging

### 3. Added Temporary Debug Checks

**Purpose:** Verify logger configuration at runtime

**Locations:**
- `src/pmarlo/reporting/plots.py` (lines 277, 306)
- `src/pmarlo/markov_state_model/_loading.py` (lines 73, 87)

**Debug Check Code:**
```python
print(f"DEBUG_CHECK: Logger name='{logger.name}', Effective level={logger.getEffectiveLevel()}, Handler count={len(logger.handlers)}, Root handler count={len(logging.getLogger().handlers)}")
```

**What This Shows:**
- Logger name (e.g., `pmarlo.reporting.plots` or `pmarlo`)
- Effective log level (10 = DEBUG, 20 = INFO, etc.)
- Number of handlers on the logger and root logger
- Helps diagnose if messages are being filtered

## Verification Checklist

### Current Logging Configuration (app.py lines 107-158)

- ✅ Root logger level: `logging.DEBUG`
- ✅ File handler level: `logging.DEBUG`
- ✅ pmarlo logger level: `logging.DEBUG` (explicitly set)
- ✅ Handler check: `root_logger.hasHandlers()` prevents duplicates
- ✅ Third-party libraries: matplotlib and PIL set to INFO (less verbose)

### Expected Logger Hierarchy

```
root (DEBUG)
├── FileHandler (DEBUG) -> app_log_YYYYMMDD_HHMMSS.log
├── matplotlib (INFO)
├── PIL (INFO)
└── pmarlo (DEBUG)
    ├── pmarlo.reporting.plots (inherits DEBUG)
    └── pmarlo.markov_state_model._loading (inherits DEBUG)
```

## Testing Instructions

### 1. Test Single Log File Creation

1. Start the Streamlit app:
   ```bash
   poetry run streamlit run example_programs/app_usecase/app/app.py
   ```

2. Check log directory:
   ```bash
   dir example_programs\app_usecase\app_outputs\app_logs\
   ```

3. **Expected:** ONE log file with current timestamp

4. Interact with the app (click buttons, change tabs)

5. Check log directory again

6. **Expected:** STILL only one log file (no new files created)

### 2. Test DEBUG Messages in Log File

1. Run an operation that processes shards (e.g., "Generate Validation Plots")

2. Open the log file in `app_outputs/app_logs/app_log_YYYYMMDD_HHMMSS.log`

3. **Look for:**
   ```
   2025-10-25 10:30:45 | DEBUG    | pmarlo.reporting.plots | Shard 0: traj length = 5000
   2025-10-25 10:30:45 | DEBUG    | pmarlo.reporting.plots | Shard 0: SKIPPED - empty trajectory
   2025-10-25 10:30:45 | DEBUG    | pmarlo.reporting.plots | Shard 1: plot_len = 1000, max_traj_length_plot = 1000
   ```

4. **Also check console output for:**
   ```
   DEBUG_CHECK: Logger name='pmarlo.reporting.plots', Effective level=10, Handler count=0, Root handler count=1
   ```

### 3. Verify Logger Levels

The DEBUG_CHECK output should show:
- **Logger name:** `pmarlo.reporting.plots` or `pmarlo.markov_state_model._loading`
- **Effective level:** `10` (which is `logging.DEBUG`)
- **Handler count:** `0` (child loggers inherit from root)
- **Root handler count:** `1` (one FileHandler)

If effective level is NOT 10:
- Check if a library or plugin is overriding logging configuration
- Verify `pmarlo_logger.setLevel(logging.DEBUG)` runs after handler check

## Troubleshooting

### If DEBUG messages still don't appear:

1. **Check logger name:**
   - If DEBUG_CHECK shows `pmarlo.markov_state_model._loading` instead of `pmarlo`
   - Update app.py to set the specific logger:
   ```python
   specific_logger = logging.getLogger('pmarlo.reporting')
   specific_logger.setLevel(logging.DEBUG)
   ```

2. **Check log file permissions:**
   - Ensure the log file is writable
   - Check file size is growing when app runs

3. **Verify handler is attached:**
   - DEBUG_CHECK should show Root handler count = 1
   - If 0, the handler didn't get added (check for errors in _configure_file_logging)

4. **Check for conflicting basicConfig:**
   - Search codebase for other `logging.basicConfig()` calls
   - These can reset the logging configuration

### If multiple log files still appear:

1. **Check handler count:**
   - DEBUG_CHECK should show Root handler count = 1
   - If > 1, handlers are being added on each rerun (singleton check not working)

2. **Verify singleton check:**
   - Add print statement before handler check:
   ```python
   print(f"LOGGING_INIT: hasHandlers={root_logger.hasHandlers()}")
   ```
   - Should print `True` on all reruns after first

3. **Check for forced reloading:**
   - Some Streamlit settings can force module reloads
   - Verify no `st.experimental_rerun()` in unexpected places

## Removal of Debug Checks

Once verified that logging is working correctly, remove the temporary DEBUG_CHECK print statements:

**Files to clean up:**
1. `src/pmarlo/reporting/plots.py` - Lines 277, 306
2. `src/pmarlo/markov_state_model/_loading.py` - Lines 73, 87

Search for: `DEBUG_CHECK:` and remove those print statements.

## Summary

✅ **Fixed:** Multiple log files - Only one file created per app session
✅ **Fixed:** Missing DEBUG messages - Converted print() to logger.debug()
✅ **Added:** Debug checks to verify logger configuration
✅ **Verified:** No syntax errors in modified files
✅ **Ready:** For testing with real app usage

## Next Steps

1. Run the app and trigger validation plot generation
2. Check console output for DEBUG_CHECK messages
3. Examine log file for DEBUG-level "SKIPPED" messages
4. If issues persist, analyze DEBUG_CHECK output to identify root cause
5. Once confirmed working, remove DEBUG_CHECK print statements

