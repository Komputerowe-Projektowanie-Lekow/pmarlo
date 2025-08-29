This is a package that is supposed to be build with poetry. It's chekced with the poetry tox for the issues and formatting with each commit.

It's supposed to be a scientific package that enables end-to-end creation of workflows to produce good quality data for the dataset and specific environment of the ai aided drug design.

It's composed of many modules that should work partly independently and used as a modules in a larger framework.

The core code is in the /src/ directory, where the pmarlo package is stored. And the example programs in the /example_programs/ are an examples of the usage of the package like it's in a different directory with the environment containing a pip installed pmarlo, not the direct usage in the package.

All of the changes should be checked with tox and pytest. After some changes in the algorithmic changes you should create some intelligent unit test suite/integration suite to test if your work didn't draw any setback in the codebase.

If you see any unintelligent design flaws you are able to make some changes to make them improved, more clear and maintanable.

You can see all the information about the package and additional more devops things in the /mdfiles/ directory and the README.md file.

Those are the commands that I used in the whole package creation, if you want you can inspire yourself in the /mdfiles/commands.md

Remember that for the tests I run "C:\Users\konrad_guest\Documents\GitHub\pmarlo>poetry run pytest" in the CMD.

For the tox I run "C:\Users\konrad_guest\Documents\GitHub\pmarlo>poetry run tox" in the CMD.
