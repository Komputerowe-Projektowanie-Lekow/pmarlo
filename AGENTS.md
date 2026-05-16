# General
1. Package manager: Poetry is the canonical way to build, run, and publish the project.
2. Quality gate: Tox orchestrates checks (formatting, static analysis, packaging sanity, and tests) and must pass on every commit.
3. Scientific aim: Provide end‑to‑end workflows to generate high‑quality datasets and environment‑specific results for computer‑aided drug design.
4. Modular design: Subsystems are decoupled where sensible so they can be used independently or together inside larger workflows.
5. Source layout: Core code lives in /src/ under the pmarlo package. Example applications live in /example_programs/ and are designed to be run as if pmarlo were installed via pip, not imported from the local source tree.
6. All of the changes should be checked with tox and pytest. After some changes in the algorithmic changes you should create some intelligent unit test suite/integration suite to test if your work didn't draw any setback in the codebase.
7. If you see any unintelligent design flaws you are able to make some changes to make them improved, more clear and maintanable.
8. You can see all the information about the package and additional more devops things in the /mdfiles/ directory and the README.md file.
9. Those are the commands that I used in the whole package creation, if you want you can inspire yourself in the /mdfiles/commands.md
10. Remember that for the tests I run "{path}\pmarlo>poetry run pytest" in the CMD.
11. For the tox I run "{path}\pmarlo>poetry run tox" in the CMD.
12. Everytime when you make a change and you finish doing what you have done i want you to do the
- After you do any change use the changelog.d directory to use the scriv tool that enables to create continous changelog development. I don't want you to create the scriv nor collect them. I do all of that, but when there is a note in the changelog.d that is not AGENTS.md, append the updates, features, removals, deprecations and stuff like that.
13. Do not create any fallbacks. All module should be there and if there isn't issue it as an error not a silent fallback.
14. When you think of implementing some function think if its already implemented somwehre and could use the standard library for it, if so prefer the standard library which is already better tested than our implementation.
15. Do not use the lazy import. Currently python is as fast as it is so we don't need that technique. Prefer the maintaiability.
16. When you run the poetry run pytest or any pytest command, NEVER USE it with -noProfile.
17. Whenever a test case is required, always use real data—never stubs or synthetically generated examples. For instance, consistently use the 3gd8-fixed PDB file rather than any other fabricated version.
18. Whenever you can't run the poetry run pytest, try it with the 'POETRY_VIRTUALENVS_CREATE=true'.
19. All commands related to Python, packages, or the project (including `python`, `pip`, `pytest`, `tox`, `mypy`, `ruff`, etc.) MUST be executed using `poetry run`. This ensures the command runs inside the project's virtual environment.
    * **WRONG:** `pip install psutil`
    * **WRONG:** `python -c "import openmm"`
    * **RIGHT:** `poetry run pip install psutil`
    * **RIGHT:** `poetry run python -c "import openmm"`
20. If `poetry run pytest` or `poetry run tox` fails with a `ModuleNotFoundError` (indicating a missing dependency):
    * **DO NOT** modify the test code to skip the test (e.g., using `pytest.importorskip`). This violates Rule #13 (No Fallbacks).
    * **DO** run the maintenance script: `./codex_maintenance_script.sh` to attempt to repair the environment and install missing dependencies.
    * After the maintenance script runs, **DO** re-run the original `poetry run pytest ...` or `poetry run tox ...` command to verify the fix.
21. Write clean and elegant code, without overcomplications, do not use emojis in the codebase.
22. When you write in the changelog, only do the sections that are from those categories, do not come up with others. Those are the categories: "added", "fixed", "changed", "removed".

# CLAUDE.md — 12-rule template

These rules apply to every task in this project unless explicitly overridden.

Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

## Rule 1 — Think Before Coding

State assumptions explicitly. If uncertain, ask rather than guess.
Present multiple interpretations when ambiguity exists.
Push back when a simpler approach exists.
Stop when confused. Name what is unclear.

## Rule 2 — Simplicity First

Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked.
No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

## Rule 3 — Surgical Changes

Touch only what you must. Clean up only your own mess.
Do not improve adjacent code, comments, or formatting.
Do not refactor what is not broken.
Match existing style.

## Rule 4 — Goal-Driven Execution

Define success criteria. Loop until verified.
Do not follow steps blindly.
Define success and iterate.
Strong success criteria let you loop independently.

## Rule 5 — Use the model only for judgment calls

Use the model for:

- classification
- drafting
- summarization
- extraction from unstructured text

Do not use the model for:

- routing
- retries
- status-code handling
- deterministic transforms

If code can answer, code answers.

## Rule 6 — Token budgets are not advisory

Per-task budget: 4,000 tokens.
Per-session budget: 30,000 tokens.

If approaching budget, summarize and start fresh.
Surface the breach. Do not silently overrun.

## Rule 7 — Surface conflicts, do not average them

If two patterns contradict, do not blend them.
Pick one, preferably the more recent or more tested pattern.
Explain why.
Flag the other for cleanup.

Average code that satisfies both conflicting patterns is usually the worst code.

## Rule 8 — Read before you write

Before adding code, read:

- the file exports
- the immediate caller
- obvious shared utilities

If you do not understand why existing code is structured the way it is, ask before adding to it.

"Looks orthogonal" is dangerous.

## Rule 9 — Tests verify intent, not just behavior

Tests must encode why the behavior matters, not just what it does.

A test like `expect(getUserName()).toBe("John")` is weak if the function can pass by returning a constant.

If a test cannot fail when business logic changes, the test is wrong.

## Rule 10 — Checkpoint after every significant step

After each significant step in a multi-step task, summarize:

- what was done
- what was verified
- what remains

Do not continue from a state you cannot describe back clearly.
If you lose track, stop and restate the current state.

## Rule 11 — Match the codebase's conventions, even if you disagree

Conformance is more important than taste inside an existing codebase.

If the codebase uses snake_case, use snake_case.
If the codebase uses class-based components, use class-based components.
If the codebase uses a specific testing style, follow that style.

If you genuinely think a convention is harmful, surface it.
Do not fork the convention silently.

## Rule 12 — Fail loud

"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any tests were skipped.
"Feature works" is wrong if the requested edge case was not verified.

If you cannot be sure something worked, say so explicitly.
Default to surfacing uncertainty, not hiding it.

---

## Project-specific rules

NOT YET DONE.
