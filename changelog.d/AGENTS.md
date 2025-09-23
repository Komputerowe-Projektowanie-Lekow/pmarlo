# Changelog Entry

## Fixed
- Fixed mypy union-attr errors in `src/pmarlo/data/aggregate.py` where dict methods were called on union types (lines 183-192)
- Fixed no-any-return error in `_unique_shard_uid` function by adding explicit type annotation for canonical_id
- Improved type safety by properly handling `getattr` return values and ensuring dict type consistency

## Technical Details
- Enhanced `_unique_shard_uid` function to handle `getattr` return values safely by checking instance type before using as dict
- Refactored info dict construction to use properly typed `source_dict` instead of accessing union types directly
- Added explicit type annotation for `canonical_id: str` to resolve mypy's no-any-return warning
