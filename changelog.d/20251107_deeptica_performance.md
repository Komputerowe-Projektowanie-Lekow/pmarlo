changed:
- CV monitoring in replica exchange now uses OpenMM native forces (CustomBondForce, CustomAngleForce, CustomTorsionForce) for feature extraction when available, eliminating redundant PyTorch model inference and providing 2x speedup in monitoring overhead.
- Model export (`export_cv_model`, `export_cv_bias_potential`) now generates an additional NN-only TorchScript model (`*_nn.pt`) alongside the full model, designed for future integration with OpenMM native feature computation.
- System builder attempts to create OpenMM native feature forces for CV monitoring, falling back gracefully to legacy approach if feature specification is unavailable.
- Feature specification is now stored in model bundle metadata to enable native force creation during system setup.

fixed:
- Redundant CV computation in replica exchange monitoring removed - feature values are now computed once via OpenMM forces instead of being recalculated through separate PyTorch model inference.
- CV monitoring initialization is more robust, with graceful fallback to legacy PyTorch monitoring if native forces are not available.

added:
- New module `pmarlo.features.deeptica.openmm_features` with utilities to create OpenMM forces from feature specifications, including `create_feature_forces()`, `load_feature_spec_from_model()`, and `extract_cv_values_from_context()`.
- Function `extract_nn_only_from_bias_module()` in `ts_feature_extractor.py` to extract neural network layers separately from feature extraction for potential future optimizations.
- Native OpenMM feature forces (force group 2) are created during system setup when feature specification is available in model metadata.
- Optimized CV monitoring path `_update_bias_monitor_native()` that uses OpenMM forces instead of PyTorch, with automatic fallback to legacy `_update_bias_monitor_pytorch()` for older models.

