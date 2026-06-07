#!/bin/bash

echo "=== inference_patas.sh started at $(date) ==="
echo "host: $(hostname)"
echo "cwd:  $(pwd)"
echo "args: $@"

source environments/.envrc
source .venv/bin/activate
uv sync
uv run python src/finetune/run_inference.py "$@"
echo "=== inference_patas.sh finished at $(date) ==="