#!/bin/bash

echo "=== medjex_patas.sh started at $(date) ==="
echo "host: $(hostname)"
echo "cwd:  $(pwd)"
echo "args: $@"

export HF_HOME=$HOME/hf-cache
mkdir -p "$HF_HOME"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medjargone-gpu

echo "python: $(which python)"
echo "conda env: $CONDA_DEFAULT_ENV"

python medjex_pipeline.py "$@"
echo "=== medjex_patas.sh finished at $(date) ==="
