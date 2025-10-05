### Fixed
- Declared stable DeepTICA trainer aliases and history helpers so numpy-derived arrays and curriculum settings satisfy the typing gate.
- Tightened demultiplexing metadata, shard ID, and transform utilities to return concrete types and align futures bookkeeping for mypy.
- Cleaned DeepTICA facade, demux stubs, and tests so flake8 passes without suppressing project defaults.
