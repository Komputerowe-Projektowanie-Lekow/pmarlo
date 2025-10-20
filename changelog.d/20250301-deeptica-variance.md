### Changed
- Compute DeepTICA output variance directly with PyTorch tensors to avoid redundant NumPy transfers and keep GPU execution paths consistent on CPU and CUDA devices.

### Added
- Unit tests covering the tensor-based variance helper to ensure biased and unbiased variance cases remain stable.
