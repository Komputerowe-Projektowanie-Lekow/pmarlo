```
poetry lock
poetry install --with dev,tests,docs --extras fixer --no-root
poetry install --with dev,tests,docs --extras fixer
```

```
git ls-remote --tags https://github.com/pre-commit/mirrors-isort
poetry run pre-commit install
poetry run pre-commit run --all-files
```

```
git add -u
poetry run pre-commit run --all-files
poetry add --group dev mypy@^1.17
```

```
poetry publish --build --repository testpypi
git tag v0.0.8
git push --tags
poetry dynamic-versioning show
```

```
poetry run scriv create
poetry run scriv collect --version 0.12.0 --add
```

## algorithms testing
```
docker build -t pmarlo-exp .
```
```
docker run --rm -it -v "${PWD}:/app" pmarlo-exp `
  python -m pmarlo.experiments.cli simulation --steps 500 `
  --pdb tests/data/3gd8-fixed.pdb
```
```
docker run --rm -it -v "${PWD}:/app" pmarlo-exp `
  python -m pmarlo.experiments.cli remd --steps 800 --equil 200 `
  --pdb tests/data/3gd8-fixed.pdb
```
```
docker run --rm -it -v "${PWD}:/app" pmarlo-exp `
  python -m pmarlo.experiments.cli msm `
  --traj tests/data/traj.dcd `
  --top  tests/data/3gd8-fixed.pdb
```

```
docker run --rm -it pmarlo-exp python -m pmarlo.experiments.cli --help
```

## need to run them before they are correct, not yet known
