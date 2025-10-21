### Fixed
- Ensured the KDE neighbour smoothing helper returns precise floats, eliminating ``Any`` propagation that broke strict typing.
- Taught `_coerce_dtrajs` to accept mapping inputs without type ambiguity so ``tox -e type`` passes on analysis debug exports.
