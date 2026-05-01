#!/bin/bash

echo "=== inference_patas.sh started at $(date) ==="
echo "host: $(hostname)"
echo "cwd:  $(pwd)"
echo "args: $@"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate medjargone-gpu

echo "python: $(which python)"
echo "conda env: $CONDA_DEFAULT_ENV"

python run_inference.py "$@"
echo "=== inference_patas.sh finished at $(date) ==="
