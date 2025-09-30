set -euxo pipefail

# 0) OS deps
sudo apt-get update
sudo apt-get install -y --no-install-recommends python3-venv python3-dev build-essential git ca-certificates

# 1) Project venv
VENV=/workspace/.venv
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -U pip setuptools wheel

# 2) Poetry (keep inside the venv; do NOT use 'poetry install --sync')
python -m pip install "poetry==2.1.3"

# 3) Make sure we're on the development branch of your repo
cd /workspace/pmarlo
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/Komputerowe-Projektowanie-Lekow/pmarlo.git
git fetch origin --tags --prune
git checkout -B development origin/development

# 4) Install CPU-only torch first to steer resolution away from CUDA wheels
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0"

# 5) Install the project + extras (Poetry uses your pyproject just fine)
poetry config virtualenvs.create false
poetry lock
poetry install --with dev,tests --extras "mlcv app fixer" --no-interaction --no-ansi

# 6) Re-pin CPU torch in case Poetry pulled a CUDA build via transitive deps
python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3.0" --upgrade

# 7) Quick smoke
python - <<'PY'
import pmarlo, torch, numpy, openmm
print("pmarlo:", getattr(pmarlo, "__version__", "unknown"))
print("torch:", torch.__version__)
print("openmm:", openmm.__version__)
print("numpy:", numpy.__version__)
print("smoke OK")
PY
