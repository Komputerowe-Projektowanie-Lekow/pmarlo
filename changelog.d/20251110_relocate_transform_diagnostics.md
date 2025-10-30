## Changed

- Redirected transform diagnostics to resolve alongside dataset or configured
  output directories instead of creating a top-level `pmarlo_diagnostics`
  folder, honouring the new ``BuildOpts.diagnostics_dir`` override and
  repository output conventions.
- Added unit coverage for diagnostics directory resolution so environment and
  example program heuristics stay exercised.
