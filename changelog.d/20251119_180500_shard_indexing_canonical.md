### changed
- Shard index discovery now enforces canonical seg/rep components in shard filenames, raising on malformed names instead of silently skipping them.

### removed
- Legacy `shard_*.json` filename handling when determining the next shard index; directories must use the canonical shard naming convention.
