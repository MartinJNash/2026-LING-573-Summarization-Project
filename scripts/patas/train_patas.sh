#!/bin/bash

echo "=== train_patas.sh started at $(date) ==="
echo "host: $(hostname)"
echo "cwd:  $(pwd)"
echo "args: $@"

source environments/.envrc
source .venv/bin/activate
uv sync
uv run python src/finetune/train.py "$@"

echo "=== train_patas.sh finished at $(date) ==="