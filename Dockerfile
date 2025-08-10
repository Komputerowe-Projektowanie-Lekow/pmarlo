# Minimal CPU-only image to run PMARLO experiments
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PMARLO=0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry and configure to install into system site-packages
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# Copy only dependency manifests first for better layer caching
COPY pyproject.toml poetry.lock LICENSE /app/

# Install dependencies from poetry.lock (skip installing project itself)
RUN poetry install --no-interaction --no-ansi --no-root

# Copy source and tests (for bundled test data)
COPY src /app/src
COPY tests /app/tests

# Make src importable without installing the package (live code via bind mount)
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "pmarlo.experiments.cli", "--help"]
