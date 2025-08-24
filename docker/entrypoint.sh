#!/usr/bin/env bash
set -euo pipefail

# If a command is passed, execute it (preserves existing usage like
# `docker run IMAGE python -m pmarlo.experiments.cli ...`).
if [[ "$#" -gt 0 ]]; then
  exec "$@"
fi

# Resolve K8s indexed Job completion index if present
JOB_INDEX_VAL="${JOB_INDEX:-}"

# If not provided explicitly, allow alternate env var
if [[ -z "${JOB_INDEX_VAL}" && -n "${K8S_JOB_COMPLETION_INDEX:-}" ]]; then
  JOB_INDEX_VAL="${K8S_JOB_COMPLETION_INDEX}"
fi

# If still not set, fall back to CLI help for local/manual runs
if [[ -z "${JOB_INDEX_VAL}" ]]; then
  exec python -m pmarlo.experiments.cli --help
fi

# Delegate to predefined benchmark suite to ensure outputs land
# in standard experiments_output/<algorithm> locations and follow
# baseline/trend conventions implemented by experiment modules.
index="${JOB_INDEX_VAL}"
exec python -m pmarlo.experiments.suite --index "${index}"
