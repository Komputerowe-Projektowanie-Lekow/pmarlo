### Fixed

- Ensure protein input paths expand user and environment variables before
  validation, allowing `Protein` to load structures referenced via `~` and
  other shell-style paths.
