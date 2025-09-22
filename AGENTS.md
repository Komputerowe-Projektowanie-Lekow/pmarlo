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
11. For the tox I run "{path}}\pmarlo>poetry run tox" in the CMD.
12. Everytime when you make a change and you finish doing what you have done i want you to do the 