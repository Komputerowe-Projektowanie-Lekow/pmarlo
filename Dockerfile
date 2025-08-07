# Minimal CPU-only image to run PMARLO experiments
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PMARLO=0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml poetry.lock requirements.txt requirements-dev.txt /app/
COPY src /app/src
COPY tests /app/tests

# Install deps (prefer requirements for speed, fall back if missing)
RUN pip install --no-cache-dir -r requirements.txt || true

# Editable install for development
RUN pip install --no-cache-dir -e /app

CMD ["python", "-m", "pmarlo.experiments.cli", "--help"]
