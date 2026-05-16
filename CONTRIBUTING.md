## Development
Try to avoid the complexity at all cost. It's currently overbloated and in a need of the refactor after it starts working correctly. So after any changes look at the LoC changes and think if those changes are really needed and couldn't be created more compacted.

1. ## General developer notes
2.
3. Install Poetry & run `poetry install --with dev`.
2. Run `pre-commit install` once.
3. Make changes; all hooks should pass before pushing.


## Example programs
Runnable examples live in ``example_programs`` and are numbered in the intended
reading order. Execute them directly with Poetry, for example
``poetry run python example_programs/01_verify_pmarlo.py``. Each example writes
to the matching numbered directory under ``example_programs/programs_outputs``.
