### added
- Added `tests/test_compute_macrostate_memberships.py` to lock down PCCA+ membership behavior and validate canonical macrostate ordering along with obvious input errors.

### fixed
- `_compute_macrostate_memberships` now rejects requests for fewer than one macrostate and enforces integer input validation.
