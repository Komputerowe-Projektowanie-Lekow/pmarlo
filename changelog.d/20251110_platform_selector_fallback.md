## fixed

- Let the replica-exchange OpenMM platform selector fall back to the first available backend when neither CUDA nor CPU are present so platform detection keeps working even with OpenCL-only installations.
