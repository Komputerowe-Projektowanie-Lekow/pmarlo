## Development
Try to avoid the complexity at all cost. It's currently overbloated and in a need of the refactor after it starts working correctly. So after any changes look at the LoC changes and think if those changes are really needed and couldn't be created more compacted.

1. ## General developer notes
2.
3. Install Poetry & run `poetry install --with dev`.
2. Run `pre-commit install` once.
3. Make changes; all hooks should pass before pushing.


## Developer utilities
Helper helpers now live either in the top-level ``scripts`` package (structural utilities)
or the ``example_programs`` package (benchmarks and diagnostics) and are executed
directly with Python. They are not part of the distributed ``pmarlo`` package.

* ``python -m example_programs.check_extras_parity`` verifies that the optional dependency
  groups declared in ``[project.optional-dependencies]`` stay in sync with
  ``[tool.poetry.extras]``.
* ``python -m scripts.lines_report`` generates the language statistics report
  based on ``pygount`` statistics.
