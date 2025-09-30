### Fixed
- Reworked DeepTICA trainer metric helpers so implicit scalars and vectors coerce to floats safely and `mypy` can verify curriculum bookkeeping.
- Normalised EnhancedMSM protocol usage across the MSM pipeline, experiments, and transforms so constructors and factories accept the documented keyword arguments during type checking.
- Tightened demultiplexing metadata helpers and transform step handler registration to return concrete types that satisfy the typing gate.
