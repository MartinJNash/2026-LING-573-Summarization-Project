#!/bin/bash

echo "=== eval_patas.sh started at $(date) ==="
echo "host: $(hostname)"
echo "cwd:  $(pwd)"
echo "args: $@"

source environments/.envrc
source .venv/bin/activate
uv sync
uv run python src/eval_pipeline.py "$@"
echo "=== eval_patas.sh finished at $(date) ==="
