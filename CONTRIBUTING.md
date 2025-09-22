1. ## General developer notes
2. 
3. Install Poetry & run `poetry install --with dev`.
2. Run `pre-commit install` once.
3. Make changes; all hooks should pass before pushing.


## Developer utilities
We ship a couple of helper commands via the ``pmarlo.devtools`` package.

* ``poetry run pmarlo-check-extras`` verifies that the optional dependency
  groups declared in ``[project.optional-dependencies]`` stay in sync with
  ``[tool.poetry.extras]``.
* ``poetry run pmarlo-lines-report`` generates the language statistics report
  that previously lived in ``utilities/lines.py``.
