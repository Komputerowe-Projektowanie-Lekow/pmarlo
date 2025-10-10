## Changed
- Optimised the reweighting kernel to reuse in-place NumPy buffers, reducing temporary allocations while validating bias lengths.

## Added
- Expanded reweighter unit coverage for TRAM aliasing, input immutability, and bias validation regressions.
