#!/bin/bash

echo "=== eval_patas.sh started at $(date) ==="
echo "host: $(hostname)"
echo "cwd:  $(pwd)"
echo "args: $@"

# Put HF cache outside home dir to avoid 20GB quota issues
export HF_HOME=$HOME/hf-cache
mkdir -p "$HF_HOME"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medjargone-gpu

echo "python: $(which python)"
echo "conda env: $CONDA_DEFAULT_ENV"

python eval_pipeline.py "$@"
echo "=== eval_patas.sh finished at $(date) ==="
