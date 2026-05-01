#!/bin/bash

echo "=== eval_patas.sh started at $(date) ==="
echo "host: $(hostname)"
echo "cwd:  $(pwd)"
echo "args: $@"

# Put HF cache outside home dir to avoid 20GB quota issues
export HF_HOME=/projects/${USER}/hf-cache
mkdir -p "$HF_HOME"

source .venv/bin/activate
echo "python: $(which python)"

python eval_pipeline.py "$@"
echo "=== eval_patas.sh finished at $(date) ==="
