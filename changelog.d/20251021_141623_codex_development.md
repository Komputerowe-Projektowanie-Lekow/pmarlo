### Added

- Added `scriv` and `tox` to the development dependency group so the lint tox
  environment and changelog tooling are available after a default install.

### Fixed

- Restored a clean `poetry run tox -e lint` run by tidying import order in
  `pmarlo.api` and removing unused test imports flagged by flake8.
