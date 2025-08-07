poetry lock
poetry install --with dev,tests,docs --extras fixer --no-root
poetry install --with dev,tests,docs --extras fixer

git ls-remote --tags https://github.com/pre-commit/mirrors-isort
poetry run pre-commit install
poetry run pre-commit run --all-files


git add -u
poetry run pre-commit run --all-files
poetry add --group dev mypy@^1.17

poetry publish --build --repository testpypi
git tag v0.0.8
git push --tags
poetry dynamic-versioning show

poetry run scriv create




## need to run them before they are correct, not yet known
poetry run scriv collect --version 0.3.0 --add
