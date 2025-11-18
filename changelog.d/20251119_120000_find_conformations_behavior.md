### added
- Added `tests/unit/conformations/test_find_conformations_behavior.py` to lock down the public `find_conformations` contract covering validation, manual-vs-auto source/sink semantics, flag interactions, and a minimal 2-state MSM scenario.

### fixed
- `find_conformations` now enforces the documented preconditions for uncertainty analysis and structure export while accurately tracking whether auto-detection was actually used, ensuring manual source/sink selections always win even when `auto_detect=True`.
