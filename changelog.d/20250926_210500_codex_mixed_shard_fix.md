### Fixed
- Prevented false mixed-kinds rejections in `pmarlo.data.aggregate` by using metadata-first shard kind inference and safe demux fallbacks when filenames lack demux hints.
- Added a regression test that exercises demux shards emitted from neutral filenames to ensure the MSM/FES builder accepts single-kind datasets.
