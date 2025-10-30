<!-- scriv release note stub -->
### Fixed
- Ensure `build_pair_info` gracefully handles multi-lag schedules with no valid
  pairs by returning empty index arrays instead of raising a ValueError.
