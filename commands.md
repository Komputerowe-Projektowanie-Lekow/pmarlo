poetry lock
poetry install --with dev,tests,docs --extras fixer --no-root
poetry install --with dev,tests,docs --extras fixer

git ls-remote --tags https://github.com/pre-commit/mirrors-isort
poetry run pre-commit install
poetry run pre-commit run --all-files

git add -u
poetry run pre-commit run --all-files
poetry add --group dev mypy@^1.17


## need to run them before they are correct, not yet known
poetry run pytest
poetry run black .
poetry run mypy .
poetry run pre-commit install
