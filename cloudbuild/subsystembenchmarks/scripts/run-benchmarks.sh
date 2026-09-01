#!/usr/bin/env bash
set -euo pipefail

source "$HOME/ssb_cloudbuild.env"
cd "$HOME/gcsfs"
source env/bin/activate

export PYTHONPATH="$HOME/gcsfs"

# Set soft file descriptor limit up to hard limit
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true

python gcsfs/tests/perf/subsystembenchmarks/run.py \
  "--group=$GROUP" \
  "--sweep-axes=$SWEEP_AXES" \
  "--filter=$FILTER" \
  "--bucket-prefix=$BUCKET_PREFIX" \
  "--bucket-type=$BUCKET_TYPE" \
  "--project=$PROJECT_ID" \
  "--location=$REGION" \
  "--zone=$ZONE" \
  "--model-id=${MODEL_ID:-}" \
  --require-amplification
