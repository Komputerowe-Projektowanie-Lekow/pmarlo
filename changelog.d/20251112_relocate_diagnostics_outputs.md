### Changed

- Redirect transform diagnostics artefacts away from the repository root so they land
  beside the originating example programs or test fixtures (falling back to the
  standard experiments output tree when no contextual location is available).
- Removed the legacy `test_output/` bundle artefact from the repository root and
  documented that generated bundles now live under `example_programs/programs_outputs/`.
